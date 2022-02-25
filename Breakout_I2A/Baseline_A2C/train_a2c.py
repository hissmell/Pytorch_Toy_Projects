import torch
import numpy as np
import ptan
import torch.optim as optim
from lib import common
from lib import models
from tensorboardX import SummaryWriter
import os

''' SETTING '''
RUN_NAME = 'Exp_02'
LEARNING_RATE = 1e-4
SAVE_OFFSET = 5.0
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda'

''' MAIN '''
if __name__ == '__main__':
    save_dir = common.set_save_path(RUN_NAME)

    envs = common.make_env()
    test_env = common.make_env(test=True)[0]
    writer = SummaryWriter(comment=f'-lr={LEARNING_RATE:8.6f}')

    net = models.A2C(envs[0].observation_space.shape,envs[0].action_space.n).to(DEVICE)
    net.load_state_dict(torch.load('C:\\Users\\82102\\PycharmProjects\\ToyProject01\\Pytorch_Toy_Projects\\Breakout_I2A\\Baseline_A2C\\save\\Exp_02\\Exp_02-frame=9197101-score=20.450000-test=319.00.pth'))
    net.to(DEVICE)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0],apply_softmax=True,device=DEVICE)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs,agent,common.GAMMA,common.REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE,eps=1e-4)

    buffer = []
    with common.RewardTracker(writer,save_offset=SAVE_OFFSET,test_env=test_env,device=DEVICE) as reward_tracker:
        with ptan.common.utils.TBMeanTracker(writer,batch_size=100) as tb_tracker:
            for frame, exp in enumerate(exp_source):
                buffer.append(exp)

                # handle new reward
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if reward_tracker.check_score(new_rewards[0],frame,net=net,save_dir=save_dir,RUN_NAME=RUN_NAME):
                        break

                if len(buffer) < common.BATCH_SIZE:
                    continue

                common.train(net,optimizer,buffer,tb_tracker,frame,DEVICE)

