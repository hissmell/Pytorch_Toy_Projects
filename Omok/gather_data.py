import os
import json
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

''' Hyper parameters, DON'T CHANGE! '''
MAX_GAME_NUM = 13000
NUM_WORKERS = 3
PLAY_EPISODE = 1
MCTS_SEARCHES = 100
MCTS_BATCH_SIZE = 8
GAMMA = 1.0

STEPS_BEFORE_TAU_0 = 5

''' Experiment setting '''
' If you change the network structure, make new experiment folder! '
EXP_NAME = 'NetV01_00' #netV[version_number]_exp_number'
iter_num = 2 # if you put None as input, last_iter_num is returned
device = 'cuda' if torch.cuda.is_available() else 'cpu'
''' Main body '''
# 멀티프로세싱으로 데이터 수집 가속
def mp_collect_experience(max_game_num,path_dict,env,local_net,name,gamma,device):
    while True:
        if len(os.listdir(path_dict['data_dir_path'])) >= max_game_num:
            break
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
                      f" Game steps : {len(os.listdir(path_dict['data_dir_path']))}"
                      f" Game length : {game_steps}\n"
                      f"------------------------------------------------------------", 'red'))
        print(colored(f"  * Used nodes in one game : {len(mcts_stores) // PLAY_EPISODE:d} \n"
                      f"  * Game speed : {step_speed:.2f} moves/sec ||"
                      f"  Calculate speed : {node_speed:.2f} node expansions/sec \n"
                      , 'cyan'))

        game_path = os.path.join(path_dict['data_dir_path'], f"game_{len(os.listdir(path_dict['data_dir_path'])):d}")
        state_list = []
        probs_list = []
        result_list = []
        for state_arr,_,probs_arr,result in reversed(game_history):
            state_list.append(state_arr)
            probs_list.append(probs_arr)
            result_list.append(result)
        state_list = np.stack(state_list,axis=0).tolist()
        probs_list = np.stack(probs_list,axis=0).tolist()
        game_data = {'states':state_list,'probs':probs_list,'results':result_list}
        with open(game_path+'.json','w') as f:
            json.dump(game_data,f)
        del mcts_stores


if __name__ == '__main__':
    path_dict,net,iter_num = common.gathering_dir_setting(exp_name=EXP_NAME
                                                          ,iter_num=iter_num
                                                          ,device=device)
    net.eval()
    env = envs.Omok(board_size=9)

    while True:
        for _ in range(PLAY_EPISODE):
            # 멀티 프로세스들마다 게임 데이터 수집
            workers = [mp.Process(target=mp_collect_experience,
                                  args=(MAX_GAME_NUM,path_dict,env,net,f"Worker{i:02d}",
                                        GAMMA,
                                        device)) for i in range(NUM_WORKERS)]

            for i in range(NUM_WORKERS):
                workers[i].start()
                time.sleep(100)
            [worker.join() for worker in workers]
            [worker.close() for worker in workers]



