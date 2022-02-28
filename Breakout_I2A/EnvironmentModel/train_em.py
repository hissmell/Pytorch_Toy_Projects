import torch
import torch.optim as optim
import numpy as np
import ptan
import time
import os

from lib import common
from lib import models

from tensorboardX import SummaryWriter
from collections import deque

''' Hyperparameters '''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN_NAME = 'Exp_05_Ver3_100_1'
LEARNING_RATE = 5e-4
SAVE_OFFSET = 0.05
MOVING_AVG_WIDTH = 1000

''' MAIN '''
if __name__ == '__main__':
    save_dir_path = common.set_save_path(RUN_NAME)

    envs = common.make_env()
    writer = SummaryWriter(comment=RUN_NAME + f'-lr={LEARNING_RATE:8.6f}')

    # 여기에 미리 훈련된 베이스라인 네트워크를 로드해주어야 합니다.
    net_act = models.A2C(envs[0].observation_space.shape,envs[0].action_space.n)
    net_act.load_state_dict(torch.load('C:\\Users\\82102\\PycharmProjects\\ToyProject01\\Pytorch_Toy_Projects'
                                       '\\Breakout_I2A\\Baseline_A2C\\save\\Exp_02'
                                       '\\Exp_02-frame=61969-score=15.141304-test=383.70.pth'))
    net_act.to(DEVICE)
    net_em = models.EnvironmentModelVer3(envs[0].observation_space.shape, envs[0].action_space.n).to(DEVICE)
    start_frame = 0
    optimizer = optim.Adam(net_em.parameters(),lr=LEARNING_RATE)

    agent = ptan.agent.PolicyAgent(model=lambda x:net_act(x)[0],device=DEVICE,apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs,agent,gamma=common.GAMMA)

    saved_frame = 0
    batch = []
    scores = []
    time_start = 0.0
    frame_start = 0
    best_loss = None
    loss_queue = deque(maxlen=MOVING_AVG_WIDTH)
    obs_loss_queue = deque(maxlen=MOVING_AVG_WIDTH)
    reward_loss_queue = deque(maxlen=MOVING_AVG_WIDTH)
    with ptan.common.utils.TBMeanTracker(writer,batch_size=100) as tb_tracker:
        for frame_idx, exp in enumerate(exp_source):
            frame = frame_idx + start_frame
            batch.append(exp)
            if len(batch) < common.BATCH_SIZE:
                continue

            loss, loss_obs, loss_rew = common.train(net_em,optimizer,batch,tb_tracker,frame,DEVICE)
            loss_queue.append(loss)
            obs_loss_queue.append(loss_obs)
            reward_loss_queue.append(loss_rew)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                scores.append(new_rewards[0])
                speed = (frame - frame_start) / (time.time() - time_start)
                print(f"Frame : {frame} | Reward MA : {np.mean(scores[-100:]):.2f} | speed : {speed:.2f} frame/sec "
                      f"| Total loss : {np.mean(loss_queue):.6f} | Loss obs : {np.mean(obs_loss_queue):.6f} | Loss reward : {np.mean(reward_loss_queue):.6f}")
                frame_start = frame
                time_start = time.time()

            ''' Save each SAVE_TERM '''
            if best_loss is None or ((best_loss - np.mean(loss_queue)) > best_loss * SAVE_OFFSET):
                if len(loss_queue) < MOVING_AVG_WIDTH:
                    continue
                best_loss = np.mean(loss_queue)
                print(f"Saved! Total loss : {best_loss:.6f} | Loss obs : {np.mean(obs_loss_queue):.6f} | Loss reward : {np.mean(reward_loss_queue):.6f}")
                save_path = os.path.join(save_dir_path,f"frame={frame}_loss={best_loss:.6f}.pth")
                torch.save(net_em.state_dict(), save_path)