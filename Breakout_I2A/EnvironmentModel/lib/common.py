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
STACK_FRAMES = 2 # 2로 고정합니다 이거 바꾸면 baseline network도 바꿔서 다시해야합니다...
REWARD_STEPS = 1 # 반드시 1이어야 합니다.
NUM_ENVS = 16
GAMMA = 0.99
BATCH_SIZE = 64
OBSERVATION_WEIGHT = 100.0
REWARD_WEIGHT = 1.0
IMAGE_SHAPE = (84,84)

''' FUNCTIONS '''

def set_save_path(RUN_NAME):
    save_path = os.path.join(os.getcwd(),'save')
    os.makedirs(save_path,exist_ok=True)
    save_path = os.path.join(os.getcwd(),'save',RUN_NAME)
    os.makedirs(save_path,exist_ok=True)
    return save_path

def make_env(test=False, clip=True):
    '''
    This function returns a list of environments (length : NUM_ENVS)
    '''
    if test:
        args = {'reward_clipping': False,
                'episodic_life': False}
    else:
        args = {'reward_clipping': clip}
    return [ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME),stack_frames=STACK_FRAMES,**args) for _ in range(NUM_ENVS)]

def unpack_batch(batch,device):
    '''
    This function works to make batch to trainable-state
    :param batch:
    :param device:
    :return:
    '''
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx,exp in enumerate(batch):
        states.append(np.array(exp.state,copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state,copy=False))
        else:
            last_states.append(np.array(exp.state, copy=False))

    states_np = np.array(states,copy=False)
    next_states_np = np.array(last_states,copy=False)

    states_var = torch.tensor(states_np,dtype=torch.float32).to(device)
    states_diff_var = torch.tensor(next_states_np[:,-1,:,:] - states_np[:,-1,:,:],dtype=torch.float32).to(device) # shape = (N,84,84)
    actions_var = torch.tensor(actions,dtype=torch.int64).to(device)
    rewards_var = torch.tensor(rewards,dtype=torch.float32).to(device)

    return states_var,states_diff_var,actions_var,rewards_var

def train(net_em,optimizer,batch,tb_tracker,frame,device):
    states_var,states_diff_var,actions_var,rewards_var = unpack_batch(batch,device)
    batch.clear()

    optimizer.zero_grad()
    predict_next_states_diff,predict_rewards = net_em(states_var,actions_var) # (N,1,84,84) (N,1)
    predict_next_states_diff = predict_next_states_diff.squeeze(1)
    predict_rewards = predict_rewards.squeeze(-1)

    loss_observation = F.mse_loss(states_diff_var,predict_next_states_diff)
    loss_reward = F.mse_loss(rewards_var,predict_rewards)

    loss_total = loss_observation * OBSERVATION_WEIGHT + loss_reward * REWARD_WEIGHT

    loss_total.backward()
    grads = np.concatenate([p.grad.data.to('cpu').flatten() for p in net_em.parameters() if p.grad is not None])
    optimizer.step()

    ''' Record Training Data '''
    tb_tracker.track('loss_observation', loss_observation.detach().to('cpu'), frame)
    tb_tracker.track('loss_reward', loss_reward.detach().to('cpu'), frame)
    tb_tracker.track('loss_total', loss_total.detach().to('cpu'), frame)
    tb_tracker.track('grad_l2', np.sqrt(np.mean(np.square(grads))), frame)
    tb_tracker.track('grad_max', np.max(np.abs(grads)), frame)
    tb_tracker.track('grad_var', np.var(grads), frame)

    return float(loss_total.detach().to('cpu').data.numpy())\
        ,float(loss_observation.detach().to('cpu').data.numpy())\
        ,float(loss_reward.detach().to('cpu').data.numpy())