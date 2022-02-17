import torch
import ptan
import numpy as np

from tensorboardX import SummaryWriter

from lib import models
from lib import wrappers

''' Hyper-Parameters '''
VERSION = 'v1'
LEARNING_RATE = 1e-4
STOP_REWARD = 480
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_FRAMES = 1e5
DEVICE = 'cpu'
TARGET_NET_SYNC = 1e3

''' MAIN '''
if __name__ == '__main__':
    writer = SummaryWriter(comment='-CartPole_' + VERSION)
    env = wrappers.cartpole_env_make(VERSION)

    # net, target_net, epsilon_tracker, selector, agent, exp_source, buffer, optimizer, reward_tracker
    net = models.DQN(env.observation_space.shape,env.action_space.n)
    target_net = models.DQN(env.observation_space.shape,env.action_space.n)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
