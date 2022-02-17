import gym
from lib import wrappers
from lib import common
from lib import models

import torch
import ptan
import time
import numpy as np

VERSION = "v1"
FPS = 60
TEST_NUMBER = 10
#FRAME_NUM = 100000
FRAME_NUM = 100000
if __name__ == '__main__':
    device = 'cpu'
    env = wrappers.cartpole_env_make(VERSION)
    model_path = f'C:\\Users\\82102\\PycharmProjects\\ReinforcementLearning\\Toy_project\\CartPole_DQN\\CartPole-v1-Dueling_frame_160000.pth'
    net = models.DQN(env.observation_space.shape,env.action_space.n)
    net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    for i in range(1,TEST_NUMBER+1):
        state = env.reset()
        total_reward = 0.0
        while True:
            start_ts = time.time()
            env.render()
            state_var = torch.tensor(np.array([state],copy = False))
            q_values = net(state_var).data.numpy()[0]
            action = np.argmax(q_values)

            state,reward,done,_ = env.step(action)
            total_reward += reward
            if done:
                break
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

        print(f"Test number {i} || Total reward : {total_reward:.2f}")