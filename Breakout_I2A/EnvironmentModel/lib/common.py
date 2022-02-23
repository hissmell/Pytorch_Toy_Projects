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
NUM_ENVS = 50
GAMMA = 0.99
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.5
STOP_REWARD = 400
MOVING_AVG_WIDTH = 100
BATCH_SIZE = 128
OBSERVATION_WEIGHT = 10.0
REWARD_WEIGHT = 1.0
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

    states_np = np.array(states,copy=False)
    next_states_np = np.array(last_states,copy=False)

    states_var = torch.tensor(states_np,dtype=torch.float32).to(device)
    states_diff_var = torch.tensor(states_np[:,-1,:,:] - next_states_np[:,-1,:,:],dtype=torch.float32).to(device) # shape = (N,84,84)
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

    assert predict_next_states_diff.size()[1:] == (84,84)
    assert rewards_var.size() == predict_rewards.size()
    loss_observation = F.mse_loss(states_diff_var,predict_next_states_diff)
    loss_reward = F.mse_loss(rewards_var,predict_rewards)

    loss_total = loss_observation * OBSERVATION_WEIGHT + loss_reward * REWARD_WEIGHT

    loss_total.backward()
    nn.utils.clip_grad_norm_(net_em.parameters(),max_norm=CLIP_GRAD)
    grads = np.concatenate([p.grad.data.to('cpu').flatten() for p in net_em.parameters if p.grad is not None])

    ''' Record Training Data '''
    tb_tracker.track('loss_observation', loss_observation.detach().to('cpu'), frame)
    tb_tracker.track('loss_reward', loss_reward.detach().to('cpu'), frame)
    tb_tracker.track('loss_total', loss_total.detach().to('cpu'), frame)
    tb_tracker.track('grad_l2', np.sqrt(np.mean(np.square(grads))), frame)
    tb_tracker.track('grad_max', np.max(np.abs(grads)), frame)
    tb_tracker.track('grad_var', np.var(grads), frame)


''' CLASSES '''
class RewardTracker:
    '''
    It used as "with RewardTracker(writer)"
    '''
    def __init__(self,writer,device='cpu',save_offset = 5.0,test_env=None):
        self.writer = writer
        self.stop_reward = STOP_REWARD
        self.moving_avg_width = MOVING_AVG_WIDTH
        self.best_reward_MA = 0.0
        self.save_offset = save_offset
        self.test_env = test_env
        self.device = device

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0.
        self.scores = []
        return self

    def __exit__(self,*args,**kwargs):
        self.writer.close()

    def check_score(self,score,frame,epsilon=None,net=None,save_dir=None,RUN_NAME=None):
        if not self.scores:
            save_to = os.path.join(save_dir, RUN_NAME + "-init.pth")
            torch.save(net.state_dict(),save_to)

        self.scores.append(score)
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



        if moving_avg > self.best_reward_MA + self.save_offset:
            self.best_reward_MA = moving_avg
            test_scores = []
            for _ in range(10):
                obs = self.test_env.reset()
                test_score = 0.0
                while True:
                    obs_var = torch.tensor(np.array([obs], copy=False)).to(self.device)
                    action_logits_var = net(obs_var)[0]
                    action_prob = F.softmax(action_logits_var, dim=1).squeeze().data.to('cpu').numpy()
                    action = np.random.choice(self.test_env.action_space.n, p=action_prob)
                    next_obs,reward,done,_ = self.test_env.step(action)
                    test_score += reward
                    if done:
                        test_scores.append(test_score)
                        test_score = 0.0
                        break
                    obs = next_obs

            test_result = np.mean(test_scores)

            save_to = os.path.join(save_dir, RUN_NAME + f"-frame={frame}-score={moving_avg:.6f}-test={test_result:.2f}.pth")
            torch.save(net.state_dict(),save_to)

        if moving_avg > self.stop_reward:
            print(colored(f"Solved in {len(self.scores)} Games!"), 'yellow')
            save_to = os.path.join(save_dir, RUN_NAME + f"-frame={frame}-solved.pth")
            torch.save(net.state_dict(), save_to)
            return True
        return False
