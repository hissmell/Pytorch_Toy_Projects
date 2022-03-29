import os
import time
import ptan
import copy
import random
import collections
from termcolor import colored

from lib import mcts,envs,models,common
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import multiprocessing as mp

NUM_WORKERS = 4
PLAY_EPISODE = 1
MCTS_SEARCHES = 100
MCTS_BATCH_SIZE = 10
REPLAY_BUFFER = 10000
LEARNING_RATE = 0.01
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000

TO_BE_BEST_NET = 0.05

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 10

# 멀티프로세싱으로 데이터 수집 가속
class Worker(mp.Process):
    def __init__(self,env,local_net,name
                 ,global_game_steps,global_train_steps
                 ,buffer_queue,global_replay_buffer_size):
        super(Worker, self).__init__()
        self.env = env
        self.local_net = copy.deepcopy(local_net)
        self.name = name
        self.global_game_steps = global_game_steps
        self.global_train_steps = global_train_steps
        self.buffer_queue = buffer_queue
        self.global_replay_buffer_size = global_replay_buffer_size

    def run(self):
        mcts_stores = mcts.MCTS(env)
        t = time.time()
        prev_nodes = len(mcts_stores)
        game_steps = 0
        game_nodes = 0
        for _ in range(PLAY_EPISODE):
            _, steps, game_history = common.play_game(env, mcts_stores, replay_buffer=None
                                                      ,net1=best_net, net2=best_net
                                                      ,steps_before_tau_0=STEPS_BEFORE_TAU_0
                                                      ,mcts_searches=MCTS_SEARCHES
                                                      ,mcts_batch_size=MCTS_BATCH_SIZE, device=device
                                                      ,render=False,return_history=True)
            game_steps += steps
            game_nodes += len(mcts_stores) - prev_nodes
            prev_nodes = len(mcts_stores)
            mcts_stores.clear()

        dt = time.time() - t
        step_speed = game_steps / dt
        node_speed = game_nodes / dt
        print(colored(f"------------------------------------------------------------\n"
                      f"(Worker : {self.name})\n"
                      f"Train steps : {self.global_game_steps.value}"
                      f" Game steps : {self.global_train_steps.value}"
                      f" Game length : {game_steps}"
                      f"------------------------------------------------------------", 'red'))
        print(colored(f"  Used nodes in one game : {game_nodes // PLAY_EPISODE:d} \n"
                      f"  Best Performance step : {best_idx} ||"
                      f"  Replay buffer size : {self.global_replay_buffer_size.value}\n"
                      f"  Game speed : {step_speed:.2f} moves/sec ||"
                      f"  Calculate speed : {node_speed:.2f} node expansions/sec \n"
                      , 'cyan'))

        with self.global_game_steps.get_lock():
            self.global_game_steps.value += 1

        for exp in game_history:
            self.buffer_queue.put(exp)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_name = "Exp_01"
    save_dir_path = os.path.join(os.getcwd(),"saves",run_name)
    os.makedirs(save_dir_path,exist_ok=True)
    writer = SummaryWriter(comment="-" + run_name)

    env = envs.Omok(board_size=9)
    net = models.Net(env.observation_space.shape,env.action_space.n).to(device)
    best_net = models.Net(env.observation_space.shape,env.action_space.n).to(device)

    optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)

    global_game_step = mp.Value('i',0)
    global_train_step = mp.Value('i',0)
    global_replay_buffer_size = mp.Value('i', 0)
    buffer_queue = mp.Queue(maxsize=1000)
    best_idx = 0
    with ptan.common.utils.TBMeanTracker(writer,batch_size=10) as tb_tracker:
        while True:
            # 데이터 수집
            workers = [Worker(env=env, local_net=best_net, name=f"Worker{i:02d}", global_game_steps=global_game_step,
                              global_train_steps=global_train_step, buffer_queue=buffer_queue,
                              global_replay_buffer_size=global_replay_buffer_size) for i in range(NUM_WORKERS)]
            [worker.start() for worker in workers]
            [worker.join() for worker in workers]
            [worker.close() for worker in workers]

            # 수집한 데이터를 replay_buffer에 저장
            while not buffer_queue.empty():
                replay_buffer.append(buffer_queue.get())
            global_replay_buffer_size.value = len(replay_buffer)

            if global_replay_buffer_size.value < MIN_REPLAY_TO_TRAIN:
                continue

            # train
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0

            for _ in range(TRAIN_ROUNDS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
                states_var = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)

                optimizer.zero_grad()
                probs_v = torch.tensor(batch_probs,dtype=torch.float32).to(device)
                values_v = torch.tensor(batch_values,dtype=torch.float32).to(device)
                out_logits_var, out_values_var = net(states_var)

                loss_value_v = F.mse_loss(out_values_var.squeeze(-1), values_v)
                loss_policy_v = -F.log_softmax(out_logits_var, dim=1) * probs_v
                loss_policy_v = loss_policy_v.sum(dim=1).mean()

                loss_v = loss_policy_v + loss_value_v
                loss_v.backward()
                optimizer.step()
                sum_loss += loss_v.item()
                sum_value_loss += loss_value_v.item()
                sum_policy_loss += loss_policy_v.item()

            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, global_train_step.value)

            # evaluate net
            if global_train_step.value % EVALUATE_EVERY_STEP == 0:
                test_result = common.evaluate_network(env,net, best_net,
                                                      search_num=MCTS_SEARCHES*2,batch_size=MCTS_BATCH_SIZE,
                                                      render=False,rounds=EVALUATION_ROUNDS, device=device)
                print("Net evaluated, test_result = %.2f" % test_result)
                writer.add_scalar("eval_result_ratio", test_result, global_train_step.value)
                if test_result > TO_BE_BEST_NET:
                    print("Net is better than cur best, sync")
                    best_net.load_state_dict(net.state_dict())
                    best_idx += 1
                    file_name = os.path.join(save_dir_path, f"best_{best_idx:03d}_{global_train_step.value:05d}"
                                                            f"_performance_{test_result:.4f}.pth")
                    torch.save(net.state_dict(), file_name)





