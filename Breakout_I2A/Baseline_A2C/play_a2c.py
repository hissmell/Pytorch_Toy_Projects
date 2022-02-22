from lib import common
from lib import models

import time
import torch
import numpy as np
import torch.nn.functional as F

''' Hyper-parameters'''
FPS = 25


if __name__ == '__main__':
    env = common.make_env(test=True)[0]
    model_path = 'C:\\Users\\82102\\PycharmProjects\\ReinforcementLearning\\Toy_project\\Breakout_I2A\\Baseline_A2C\\save\\Exp_02\\Exp_02-frame=9464953-score=10.060000.pth'
    net = models.A2C(env.observation_space.shape,env.action_space.n)
    net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    state = env.reset()
    total_reward = 0.0
    while True:
        start_ts = time.time()
        env.render()
        state_var = torch.tensor(np.array([state],copy = False))
        action_logits_var = net(state_var)[0]
        action_prob = F.softmax(action_logits_var,dim=1).squeeze().data.to('cpu').numpy()
        action = np.random.choice(env.action_space.n,p=action_prob)

        state,reward,done,_ = env.step(action)
        total_reward += reward
        if done:
            break
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)

    print(f"Total reward : {total_reward:.2f}")