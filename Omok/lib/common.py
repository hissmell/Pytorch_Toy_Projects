import torch
import numpy as np
import copy
from lib import mcts
from lib import envs
from lib import models

from termcolor import colored
import multiprocessing as mp

import os

def play_game(env,mcts_stores,replay_buffer,net1,net2
              ,steps_before_tau_0,mcts_searches,mcts_batch_size
              ,net1_plays_first=False,device='cpu',render=False
              ,return_history=False,gamma=1.0):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]
    elif isinstance(mcts_stores,mcts.MCTS):
        mcts_stores = [mcts_stores,mcts_stores]

    state = env.reset()
    nets = [net1.to(device),net2.to(device)]

    if not net1_plays_first:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0
    if cur_player == 0:
        net1_color = -1
    else:
        net1_color = 1

    step = 0
    tau = 0.08 if steps_before_tau_0 <= step else 1
    game_history = []

    result = None
    net1_result = None
    turn = -1 # (-1) represents Black turn! (1 does White turn)
    while result is None:
        if render:
            print(env.render())

        mcts_stores[cur_player].update_root_node(env,state,nets[cur_player],device=device)
        mcts_stores[cur_player].search_batch(mcts_searches,mcts_batch_size
                                             ,state,turn,nets[cur_player]
                                             ,device=device)
        probs,_ = mcts_stores[cur_player].get_policy_value(state,tau=tau)
        game_history.append((state,cur_player,probs))
        action = int(np.random.choice(mcts_stores[0].all_action_list, p=probs))

        if render:
            if turn == -1:
                if net1_color == -1:
                    print(colored(f"Turn : Net 1 (O turn) Nodes {len(mcts_stores[cur_player].probs):}",'blue'))
                else:
                    print(colored(f"Turn : Net 2 (O turn) Nodes {len(mcts_stores[cur_player].probs):}", 'blue'))
            else:
                if net1_color == -1:
                    print(colored(f"Turn : Net 2 (X turn) Nodes {len(mcts_stores[cur_player].probs):}",'blue'))
                else:
                    print(colored(f"Turn : Net 1 (X turn) Nodes {len(mcts_stores[cur_player].probs):}", 'blue'))
            N_dict, Q_dict = mcts_stores[cur_player].get_root_child_statistics()
            top = min(3, len(env.legal_actions))
            N_list = sorted(list(N_dict.keys()), key=lambda x: N_dict[x], reverse=True)
            for i in range(1, top + 1):
                print(colored(
                    f'Top {i} Action : ({N_list[i - 1][0]:d},{N_list[i - 1][1]:d})'
                    f' Visit : {N_dict[N_list[i - 1]]} Q_value : {Q_dict[N_list[i - 1]]:.3f}'
                    f' Prob : {probs[env.encode_action(N_list[i - 1])]*100:.2f} %','cyan'))
            move = env.decode_action(action)
            print(colored(f"Action taken : ({move[0]:d},{move[1]:d})"
                          f" Visit : {N_dict[move]} Q_value : {Q_dict[move]:.3f}"
                          f" Prob : {probs[env.encode_action(move)]*100:.2f} %",'red'))


        state,reward,done,_ = env.step(action)
        if done:
            if render:
                print(env.render())
            result = reward
            if net1_color == -1:
                net1_result = reward
            else:
                net1_result = -reward

        cur_player = 1 - cur_player
        turn = -turn

        step += 1
        if step >= steps_before_tau_0:
            tau = 0.08


    h = []
    if replay_buffer is not None or return_history:
        for state, cur_player, probs in reversed(game_history):
            if replay_buffer is not None:
                replay_buffer.append((state, cur_player, probs, result))
            if return_history:
                h.append((copy.deepcopy(state), cur_player, probs, result))

            result = -result * gamma

    return net1_result, step, h

def evaluate_network(env,net1,net2,rounds,search_num=200,batch_size=10,device='cpu',render=False):
    net1_score = 0
    mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]
    for round in range(rounds):
        result,_,_ = play_game(env,mcts_stores,None,net1,net2,0
                             ,search_num,batch_size,False,device
                             ,render,return_history=False)
        net1_score += result

    return net1_score / rounds

def load_model(load_path=None,device='cpu'):
    env = envs.Omok(9)
    net = models.NetV2()
    if not load_path is None:
        net.load_state_dict(torch.load(load_path,map_location=device))
    return net

def render_history(env,history):
    game_length = len(history)
    print(f"Game length : {game_length:d}")
    for step,(state,cur_player,probs,result) in enumerate(history):
        print(f"Step : {step} (Current player {cur_player})")
        print(f"Reward : {result}")
        print("Probability :")
        print(probs)
        print('Top 1 action :',env.decode_action(np.argmax(probs)),f'{probs[np.argmax(probs)]*100:.2f}%')
        print(env.render_observation(state))

def render_game_data(game_data):
    game_history = []
    for i in range(len(game_data['states'])):
        exp = (np.array(game_data['states'][i]), None, np.array(game_data['probs'][i]), game_data['rewards'][i])
        game_history.append(exp)
    render_history(envs.Omok(9),game_history)

def gathering_dir_setting(exp_name,iter_num = None,device = 'cpu'):
    version = int(exp_name.split('_')[0][-2:])
    if version not in [0,1]:
        print('Version you entered :',version)
        raise Exception('network version you entered is not supplied.')

    env = envs.Omok(9)
    env.reset()
    if version == 0:
        net = models.Net(env.observation_space.shape,env.action_space.n).to(device)
    elif version == 1:
        net = models.NetV2().to(device)

    exp_dir_path = os.path.join(os.getcwd(),exp_name)
    os.makedirs(exp_dir_path,exist_ok=True)


    if iter_num == None or iter_num == 0:
        iter_num = len(os.listdir(exp_dir_path))
        iter_num = max(0,iter_num-1)
        print('Current_iteration_number :',iter_num)
    else:
        past_iter_dir_path = os.path.join(exp_dir_path,f"iter_{iter_num-1:02d}")
        net.load_state_dict(torch.load(os.path.join(past_iter_dir_path,'net_path.pth'),map_location=device))

    iter_dir_path = os.path.join(exp_dir_path,f"iter_{iter_num:02d}")
    data_dir_path = os.path.join(iter_dir_path,'raw_data')

    path_dict = {}
    path_dict['iter_dir_path'] = iter_dir_path
    path_dict['data_dir_path'] = data_dir_path
    os.makedirs(iter_dir_path, exist_ok=True)
    os.makedirs(data_dir_path, exist_ok=True)

    return path_dict,net,iter_num

def preprocessing_dir_setting(exp_name,iter_num=None,device='cpu'):
    exp_dir_path = os.path.join(os.getcwd(), exp_name)
    os.makedirs(exp_dir_path, exist_ok=True)

    if iter_num == None:
        iter_num = len(os.listdir(exp_dir_path))
        iter_num = max(0, iter_num - 1)
        print('Current_iteration_number :', iter_num)

    iter_dir_path = os.path.join(exp_dir_path, f"iter_{iter_num:02d}")
    data_dir_path = os.path.join(iter_dir_path, 'raw_data')

    path_dict = {}
    path_dict['iter_dir_path'] = iter_dir_path
    path_dict['data_dir_path'] = data_dir_path
    os.makedirs(iter_dir_path, exist_ok=True)
    os.makedirs(data_dir_path, exist_ok=True)

    return path_dict,iter_num

def training_dir_setting(exp_name,iter_num=None,device='cpu'):
    version = int(exp_name.split('_')[0][-2:])
    if version not in [0, 1]:
        print('Version you entered :', version)
        raise Exception('network version you entered is not supplied.')

    env = envs.Omok(9)
    env.reset()
    if version == 0:
        net = models.Net(env.observation_space.shape, env.action_space.n).to(device)
    elif version == 1:
        net = models.NetV2().to(device)

    exp_dir_path = os.path.join(os.getcwd(), exp_name)

    if iter_num == None or iter_num == 0:
        iter_num = len(os.listdir(exp_dir_path))
        iter_num = max(0, iter_num - 1)
        print('Current_iteration_number :', iter_num)
    else:
        past_iter_dir_path = os.path.join(exp_dir_path, f"iter_{iter_num - 1:02d}")
        net.load_state_dict(torch.load(os.path.join(past_iter_dir_path,'net_path.pth'),map_location=device))

    iter_dir_path = os.path.join(exp_dir_path, f"iter_{iter_num:02d}")
    data_dir_path = os.path.join(iter_dir_path, 'dataset.json')
    net.train()
    net.to(device)
    net_save_path = os.path.join(iter_dir_path,'net_path.pth')

    record_dir_path = os.path.join(iter_dir_path,"training_records")

    path_dict = {}
    path_dict['iter_dir_path'] = iter_dir_path
    path_dict['data_dir_path'] = data_dir_path
    path_dict['net_save_path'] = net_save_path
    path_dict['record_dir_path'] = record_dir_path

    return path_dict, net, iter_num

if __name__ == '__main__':
    from models import Net
    from mcts import MCTS
    from envs import Omok
    env = Omok(board_size=9)
    mcts_stores = MCTS(env)
    net1 = Net(env.observation_space.shape,env.action_space.n)
    net2 = Net(env.observation_space.shape,env.action_space.n)
    device = 'cuda'
    # render = True
    # net1_result,step = play_game(env,mcts_stores,None,net1,net2,0,mcts_searches=100,mcts_batch_size=10,
    #                         net1_plays_first=False,device=device,render=render)
    #
    # print(net1_result,step)

    print(evaluate_network(env,net1,net2,20,search_num=5,batch_size=5,device=device,render=False))