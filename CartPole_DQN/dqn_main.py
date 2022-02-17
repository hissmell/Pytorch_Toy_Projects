import torch
import ptan
import numpy as np

from tensorboardX import SummaryWriter

from lib import common
from lib import models
from lib import wrappers

''' Hyper-Parameters '''
VERSION = 'v1'
MODE = 'Dueling'
LEARNING_RATE = 1e-4
STOP_REWARD = 480
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_FRAMES = 1e5
DEVICE = 'cpu'
TARGET_NET_SYNC = 1e2
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1e4
BATCH_SIZE = 32
RENDER = False
SAVE_FRAMES = int(1e4)

''' MAIN '''
if __name__ == '__main__':
    writer = SummaryWriter(comment='-CartPole_' + VERSION + '-'+MODE)
    env = wrappers.cartpole_env_make(VERSION)

    # net, target_net, epsilon_tracker, selector, agent, exp_source, buffer, optimizer, reward_tracker
    net = models.DQN(env.observation_space.shape,env.action_space.n)
    target_net = models.DQN(env.observation_space.shape,env.action_space.n)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    epsilon_tracker = common.EpsilonTracker(selector,EPSILON_START,EPSILON_END,EPSILON_FRAMES)

    agent = ptan.agent.DQNAgent(net,selector,device=DEVICE)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env,agent,GAMMA,steps_count=1)
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source,buffer_size=int(REPLAY_BUFFER_SIZE))

    optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
    frame_index = 0

    with common.RewardTracker(writer,STOP_REWARD) as reward_tracker:
        while True:
            if RENDER:
                env.render()
            frame_index += 1
            exp_buffer.populate(1)
            epsilon_tracker.frame(frame_index)

            new_reward = exp_source.pop_total_rewards()
            if new_reward:
                if reward_tracker.reward(new_reward[0],frame_index,selector.epsilon):
                    break

            if len(exp_buffer) < REPLAY_BUFFER_SIZE:
                continue

            optimizer.zero_grad()
            batch = exp_buffer.sample(BATCH_SIZE)
            loss_var = common.calc_loss(batch,net,target_net,gamma=GAMMA,device=DEVICE)
            loss_var.backward()
            optimizer.step()

            if frame_index % TARGET_NET_SYNC == 0:
                target_net.load_state_dict(net.state_dict())
            if frame_index % SAVE_FRAMES == 0:
                torch.save(net.state_dict(), 'CartPole-' + VERSION +'-'+MODE+ f'_frame_{frame_index}.pth')

    torch.save(net.state_dict(), 'CartPole-' + VERSION + '-'+MODE+f'-Solved.pth')


