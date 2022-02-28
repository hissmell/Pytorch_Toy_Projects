import gym
import numpy as np
from lib import common


env = common.make_env()[0]

prev_obs = env.reset()
for idx in range(10):
    action = env.action_space.sample()
    obs,_,_,_ = env.step(action)
    if idx == 3:
        break
    prev_obs = obs

print(np.mean(np.square(obs[-1] - prev_obs[-1])))