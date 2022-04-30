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

NUM_WORKERS = 2
PLAY_EPISODE = 1
MCTS_SEARCHES = 100
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER = 8000
LEARNING_RATE = 0.005
BATCH_SIZE = 128
TRAIN_ROUNDS = 1
MIN_REPLAY_TO_TRAIN = 2000
TO_BE_BEST_NET = 0.05
GAMMA = 0.93

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 10
STEPS_BEFORE_TAU_0 = 5

# 멀티프로세싱으로 데이터 수집 가속
def mp_collect_experience(env,local_net,name,gamma
             ,global_game_steps,global_train_steps
             ,buffer_queue,global_replay_buffer_size
             ,best_idx
             ,device='cpu'):
    mcts_stores = mcts.MCTS(env)
    t = time.time()

    _, game_steps, game_history = common.play_game(env, mcts_stores, replay_buffer=None
                                              ,net1=local_net, net2=local_net
                                              ,steps_before_tau_0=STEPS_BEFORE_TAU_0
                                              ,mcts_searches=MCTS_SEARCHES
                                              ,mcts_batch_size=MCTS_BATCH_SIZE, device=device
                                              ,render=False,return_history=True,gamma=gamma)
    dt = time.time() - t
    step_speed = game_steps / dt
    node_speed = len(mcts_stores) / dt
    print(colored(f"------------------------------------------------------------\n"
                  f"(Worker : {name})\n"
                  f"Train steps : {global_train_steps.value}"
                  f" Game steps : {global_game_steps.value}"
                  f" Game length : {game_steps}\n"
                  f"------------------------------------------------------------", 'red'))
    print(colored(f"  * Used nodes in one game : {len(mcts_stores) // PLAY_EPISODE:d} \n"
                  f"  * Best Performance step : {best_idx} ||"
                  f"  Replay buffer size : {global_replay_buffer_size.value}\n"
                  f"  * Game speed : {step_speed:.2f} moves/sec ||"
                  f"  Calculate speed : {node_speed:.2f} node expansions/sec \n"
                  , 'cyan'))

    with global_game_steps.get_lock():
        global_game_steps.value += 1

    for exp in game_history:
        buffer_queue.put(exp)
    del mcts_stores

def mp_evaluate_network(env,net, best_net,global_net1_score,search_num,batch_size,render,rounds, device):
    net1_score = 0.0
    for round in range(rounds):
        mcts_stores = [mcts.MCTS(env), mcts.MCTS(env)]
        result,_,_ = common.play_game(env,mcts_stores,None,net,best_net,0
                                     ,search_num,batch_size,False,device
                                     ,render,return_history=False)
        net1_score += result
        del mcts_stores
    global_net1_score.value += net1_score / rounds


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

    best_idx = 0
    with ptan.common.utils.TBMeanTracker(writer,batch_size=10) as tb_tracker:
        while True:
            for _ in range(PLAY_EPISODE):
                # 멀티 프로세스들마다 게임 데이터 수집
                buffer_queue = mp.Manager().Queue()
                workers = [mp.Process(target=mp_collect_experience,
                                      args=(env,best_net,f"Worker{i:02d}",
                                            GAMMA,
                                            global_game_step,
                                            global_train_step,
                                            buffer_queue,
                                            global_replay_buffer_size,
                                            best_idx,
                                            device)) for i in range(NUM_WORKERS)]

                [worker.start() for worker in workers]
                [worker.join() for worker in workers]
                [worker.close() for worker in workers]

                # 수집한 데이터들을 replay_buffer로 저장
                while not buffer_queue.empty():
                    replay_buffer.append(buffer_queue.get())

            global_replay_buffer_size.value = len(replay_buffer)
            if global_replay_buffer_size.value < MIN_REPLAY_TO_TRAIN:
                continue

            # train
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0
            grad_l2 = 0.0
            grad_max = 0.0
            grad_var = 0.0

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

                grads = np.concatenate([p.grad.to('cpu').data.numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                grad_l2 += np.sqrt(np.mean(np.square(grads)))
                grad_max += np.max(np.abs(grads))
                grad_var += np.var(grads)

            global_train_step.value += 1
            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("grad_l2", grad_l2 / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("grad_max", grad_max / TRAIN_ROUNDS, global_train_step.value)
            tb_tracker.track("grad_var", grad_var / TRAIN_ROUNDS, global_train_step.value)

            # evaluate net
            if global_train_step.value % EVALUATE_EVERY_STEP == 0:
                global_net1_score = mp.Value('d',0.0)
                workers = [mp.Process(target=mp_evaluate_network,
                                      args=(env,net, best_net
                                            ,global_net1_score
                                            ,MCTS_SEARCHES*2
                                            ,MCTS_BATCH_SIZE
                                            ,False
                                            ,EVALUATION_ROUNDS
                                            ,device)) for _ in range(NUM_WORKERS)]
                [worker.start() for worker in workers]
                [worker.join() for worker in workers]
                [worker.close() for worker in workers]

                test_result = global_net1_score.value / NUM_WORKERS
                print("Net evaluated, test_result = %.2f" % test_result)
                writer.add_scalar("eval_result_ratio", test_result, global_train_step.value)
                if test_result > TO_BE_BEST_NET:
                    print("Net is better than cur best, sync")
                    best_net.load_state_dict(net.state_dict())
                    best_idx += 1
                    file_name = os.path.join(save_dir_path, f"best_{best_idx:03d}_{global_train_step.value:05d}"
                                                            f"_performance_{test_result:.4f}.pth")
                    torch.save(net.state_dict(), file_name)
