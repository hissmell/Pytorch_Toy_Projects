import torch
import numpy as np
import ptan
import gym
import os
import time

import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored

''' HYPER-PARAMETERS '''
ENV_NAME = "BreakoutNoFrameskip-v4"
STACK_FRAMES = 2
REWARD_STEPS = 5
NUM_ENVS = 50
GAMMA = 0.99
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1
STOP_REWARD = 400
MOVING_AVG_WIDTH = 10

''' FUNCTIONS '''

def set_save_path():
    save_path = os.path.join(os.getcwd(),'save')
    os.makedirs(save_path,exist_ok=True)
    return save_path

def make_env():
    '''
    This function returns a list of environments (length : NUM_ENVS)
    '''
    return [ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME),stack_frames=STACK_FRAMES) for _ in range(NUM_ENVS)]

def unpach_batch(batch,net,device):
    '''
    This function works to make batch to trainable-state
    :param batch:
    :param net:
    :param device:
    :return:
    '''
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx,exp in range(batch):
        states.append(np.array(exp.state,copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        if not exp.done:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state,copy=False))

    states_var = torch.tensor(np.array(states,copy=False),dtype=torch.float32).to(device)
    actions_var = torch.tensor(actions,dtype=torch.int64).to(device)
    rewards_np = np.array(rewards,dtype=np.float32)

    if not_done_idx:
        last_states_var = torch.tensor(np.array(last_states),dtype=torch.float32)
        values_var = net(last_states_var)[1][:,0] # to 1-D array
        rewards_np[not_done_idx] += values_var.data.to('cpu').numpy() * (GAMMA ** REWARD_STEPS)

    value_refer_var = torch.tensor(rewards_np,dtype=torch.float32).to(device)
    return states_var,actions_var,value_refer_var

''' CLASSES '''
class RewardTracker:
    '''
    It used as "with RewardTracker(writer)"
    '''
    def __init__(self,writer):
        self.writer = writer
        self.stop_reward = STOP_REWARD
        self.moving_avg_width = MOVING_AVG_WIDTH

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0.
        self.scores = []

    def check_score(self,score,frame,epsilon=None):
        self.scores = []
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts = time.time()
        self.ts_frame = frame

        moving_avg = np.mean(self.scores[-self.moving_avg_width:])
        epsilon_str = '' if epsilon is None else f", Epsilon : {epsilon:.3f}"
        template = colored(f"Frame {frame} ({len(self.scores)} games) || Reward_MA{self.moving_avg_width} : {moving_avg:.2f}"
                           f" Speed : {speed:.2f} frame/sec" + epsilon_str, 'cyan')
        print(template)

        self.writer.add_scalar(f'score_moving_avg{self.moving_avg_width}', moving_avg, frame)
        self.writer.add_scalar('speed', speed, frame)
        self.writer.add_scalar('score', score, frame)

        if moving_avg > self.stop_reward:
            print(colored(f"Solved in {len(self.scores)} Games!"), 'yellow')
            return True
        return False
