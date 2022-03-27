import os
import time
import ptan
import random
import collections
from termcolor import colored

from lib import mcts,envs,models,common
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F

PLAY_EPISODE = 1
MCTS_SEARCHES = 100
MCTS_BATCH_SIZE = 10
REPLAY_BUFFER = 5000
LEARNING_RATE = 0.01
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000

TO_BE_BEST_NET = 0.05

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 50
STEPS_BEFORE_TAU_0 = 10

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
    mcts_stores = mcts.MCTS(env)

    step_idx = 0
    best_idx = 0
    with ptan.common.utils.TBMeanTracker(writer,batch_size=10) as tb_tracker:
        while True:
            t = time.time()
            prev_nodes = len(mcts_stores)
            game_steps = 0
            game_nodes = 0
            for _ in range(PLAY_EPISODE):
                _, steps = common.play_game(env,mcts_stores, replay_buffer,net1=best_net,net2=best_net,
                                           steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                                           mcts_batch_size=MCTS_BATCH_SIZE, device=device,render=False)
                game_steps += steps
                game_nodes += len(mcts_stores) - prev_nodes
                prev_nodes = len(mcts_stores)
                mcts_stores.clear()

            dt = time.time() - t
            step_speed = game_steps / dt
            node_speed = game_nodes / dt
            tb_tracker.track("Step_speed",step_speed,step_idx)
            tb_tracker.track("node_speed",node_speed,step_idx)
            print(colored(f"------------------------------------------------------------\n"
                          f"Train steps : {step_idx} \n Game steps : {game_steps}\n"
                          f"------------------------------------------------------------\n",'red'))
            print(colored(f"  Used nodes in one game : {game_nodes//PLAY_EPISODE:d} \n"
                          f"  Best Performance step : {best_idx} ||"
                          f"  Replay buffer size : {len(replay_buffer)}\n"
                          f"  Game speed : {step_speed:.2f} moves/sec ||"
                          f"  Calculate speed : {node_speed:.2f} node expansions/sec \n"
                          ,'cyan'))
            step_idx += 1
            if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
                continue

            # train
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0

            for _ in range(TRAIN_ROUNDS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
                states_var = torch.tensor(batch_states, dtype=torch.float32)

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

            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

            # evaluate net
            if step_idx % EVALUATE_EVERY_STEP == 0:
                test_result = common.evaluate_network(env,net, best_net,
                                                      render=False,rounds=EVALUATION_ROUNDS, device=device)
                print("Net evaluated, test_result = %.2f" % test_result)
                writer.add_scalar("eval_result_ratio", test_result, step_idx)
                if test_result > TO_BE_BEST_NET:
                    print("Net is better than cur best, sync")
                    best_net.load_state_dict(net.state_dict())
                    best_idx += 1
                    file_name = os.path.join(save_dir_path, f"best_{best_idx:03d}_{step_idx:05d}.pth")
                    torch.save(net.state_dict(), file_name)





