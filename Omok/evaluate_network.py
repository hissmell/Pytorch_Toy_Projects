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

def mp_evaluate_network(name,env,net1, net2
                        ,global_net1_win
                        ,global_net1_draw
                        ,global_net1_lose
                        ,search_num,batch_size,render,rounds, device):
    net1_score = 0.0
    render = True if name == "Worker_0" else False
    for round in range(rounds):
        mcts_stores = [mcts.MCTS(env), mcts.MCTS(env)]
        result,_,_ = common.play_game(env,mcts_stores,None,net1,net2,0
                                     ,search_num,batch_size,False,device
                                     ,render,return_history=False)
        if result == 1:
            global_net1_win.value += 1
        elif result == 0:
            global_net1_draw.value += 1
        elif result == -1:
            global_net1_lose.value += 1
        else:
            raise ValueError("???")
        del mcts_stores



if __name__ == '__main__':
    env = envs.Omok(board_size=9)
    global_net1_win = mp.Value('d',0.0)
    global_net1_draw = mp.Value('d',0.0)
    global_net1_lose = mp.Value('d',0.0)

    net1_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01\\' \
               'Pytorch_Toy_Projects\\Omok\\NetV01_00\\iter_02\\net_path.pth'
    net2_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01\\' \
               'Pytorch_Toy_Projects\\Omok\\NetV01_00\\iter_00\\net_path.pth'

    # net2_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01\\' \
    #        'Pytorch_Toy_Projects\\Omok\\NetV01_00\\iter_00\\net_path.pth'
    net1 = common.load_model(load_path=net1_path,device='cuda')
    net2 = common.load_model(load_path=net2_path,device='cuda')

    workers = [mp.Process(target=mp_evaluate_network,
                          args=(f"Worker_{i:d}",env,net1, net2
                                ,global_net1_win
                                ,global_net1_draw
                                ,global_net1_lose
                                ,100*2
                                ,8
                                ,False
                                ,7
                                ,'cuda')) for i in range(3)]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]
    [worker.close() for worker in workers]


    print(f"Net evaluated, Win : {global_net1_win.value:.2f}"
          f"  Draw : {global_net1_draw.value:.2f}"
          f"  Lose : {global_net1_lose.value:.2f}")