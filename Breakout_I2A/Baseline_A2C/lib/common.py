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
STACK_FRAMES = 2
REWARD_STEPS = 5
NUM_ENVS = 50
GAMMA = 0.99
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.5
STOP_REWARD = 400
MOVING_AVG_WIDTH = 100
BATCH_SIZE = 128

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

def unpack_batch(batch,net,device):
    '''
    This function works to make batch to trainable-state
    :param batch:
    :param net:
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

    states_var = torch.tensor(np.array(states,copy=False),dtype=torch.float32).to(device)
    actions_var = torch.tensor(actions,dtype=torch.int64).to(device)
    rewards_np = np.array(rewards,dtype=np.float32)

    if not_done_idx:
        last_states_var = torch.tensor(np.array(last_states),dtype=torch.float32).to(device)
        values_var = net(last_states_var)[1][:,0] # to 1-D array
        rewards_np[not_done_idx] += values_var.data.to('cpu').numpy() * (GAMMA ** REWARD_STEPS)

    value_refer_var = torch.tensor(rewards_np,dtype=torch.float32).to(device)
    return states_var,actions_var,value_refer_var

def train(net,optimizer,batch,tb_tracker,frame,device):
    states_var,actions_var,value_refer_var = unpack_batch(batch,net,device)
    batch.clear()

    optimizer.zero_grad()
    logits_var,values_var = net(states_var)

    loss_value = F.mse_loss(values_var.squeeze(-1),value_refer_var)

    log_prob_var = F.log_softmax(logits_var,dim=1)
    advance_var = value_refer_var - values_var.squeeze(1).detach()
    log_prob_action_var = log_prob_var[range(BATCH_SIZE),actions_var]
    loss_policy = advance_var.detach() * log_prob_action_var
    loss_policy = -loss_policy.mean()

    prob_var = F.softmax(logits_var,dim=1)
    loss_entropy = log_prob_var * prob_var
    loss_entropy = ENTROPY_BETA * loss_entropy.sum(dim=1).mean()

    loss_value_policy = loss_policy + loss_value
    loss_value_policy.backward(retain_graph=True)
    grads = np.concatenate([p.grad.data.to('cpu').numpy().flatten() for p in net.parameters() if p.grad is not None])

    loss_entropy.backward()
    nn.utils.clip_grad_norm_(net.parameters(),CLIP_GRAD)

    optimizer.step()
    loss_total = loss_value_policy + loss_entropy

    ''' Record Training Data '''
    tb_tracker.track('advantage', advance_var.detach().mean().to('cpu'), frame)
    tb_tracker.track('values', values_var.detach().mean().to('cpu'), frame)
    tb_tracker.track('batch_rewards', value_refer_var.detach().mean().to('cpu'), frame)
    tb_tracker.track('loss_entropy', loss_entropy.detach().to('cpu'), frame)
    tb_tracker.track('loss_policy', loss_policy.detach().to('cpu'), frame)
    tb_tracker.track('loss_value', loss_value.detach().to('cpu'), frame)
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
