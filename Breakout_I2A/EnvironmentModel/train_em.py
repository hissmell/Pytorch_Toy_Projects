import torch
import torch.optim as optim
import numpy as np
import ptan
import time
import os

from lib import common
from lib import models

from tensorboardX import SummaryWriter

''' Hyperparameters '''
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda'
RUN_NAME = 'Exp_01'
LEARNING_RATE = 5e-4
SAVE_OFFSET = 0.05

''' MAIN '''
if __name__ == '__main__':

    save_dir_path = common.set_save_path(RUN_NAME)

    envs = common.make_env()
    writer = SummaryWriter(comment=f'-lr={LEARNING_RATE:8.6f}')

    net_act = models.A2C(envs[0].observation_space.shape,envs[0].action_space.n)
    net_act.load_state_dict(torch.load('C:\\Users\\82102\\PycharmProjects\\ToyProject01\\Pytorch_Toy_Projects'
                                       '\\Breakout_I2A\\Baseline_A2C\\save\\Exp_02'
                                       '\\Exp_02-frame=61969-score=15.141304-test=383.70.pth'))
    net_act.to(DEVICE)
    net_em = models.EnvironmentModel(envs[0].observation_space.shape,envs[0].action_space.n).to(DEVICE)
    optimizer = optim.Adam(net_em.parameters(),lr=LEARNING_RATE)

    agent = ptan.agent.PolicyAgent(model=lambda x:net_act(x)[0],device=DEVICE,apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs,agent,gamma=common.GAMMA)

    saved_frame = 0
    batch = []
    scores = []
    time_start = 0.0
    frame_start = 0
    best_loss = None
    with ptan.common.utils.TBMeanTracker(writer,batch_size=100) as tb_tracker:
        for frame, exp in enumerate(exp_source):
            batch.append(exp)
            if len(batch) < common.BATCH_SIZE:
                continue

            loss, loss_obs, loss_rew = common.train(net_em,optimizer,batch,tb_tracker,frame,DEVICE)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                scores.append(new_rewards[0])
                speed = (frame - frame_start) / (time.time() - time_start)
                print(f"Frame : {frame} | Reward MA : {np.mean(scores[-100:]):.2f} | speed : {speed:.2f} frame/sec "
                      f"| Total loss : {loss:.4f} | Loss obs : {loss_obs:.4f} | Loss reward : {loss_rew:.4f}")
                frame_start = frame
                time_start = time.time()

            ''' Save each SAVE_TERM '''
            if best_loss is None or ((best_loss - loss) > best_loss * SAVE_OFFSET):
                best_loss = loss
                print(f"Saved! Total loss : {loss:.4f} | Loss obs : {loss_obs:.4f} | Loss reward : {loss_rew:.4f}")
                save_path = os.path.join(save_dir_path,f"frame={frame}_loss={loss:.4f}")
                torch.save(net_em.state_dict(), save_path)