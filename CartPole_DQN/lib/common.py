from termcolor import colored
import time
import numpy as np
import torch
import torch.nn as nn

class EpsilonTracker:
    def __init__(self,epsilon_greedy_selector,epsilon_start,epsilon_end,epsilon_frames):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_frames = epsilon_frames
        self.selector = epsilon_greedy_selector

    def frame(self,frame):
        self.selector.epsilon = max(self.epsilon_end,self.epsilon_start - frame / self.epsilon_frames)

class RewardTracker:
    def __init__(self,writer,stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward


    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self,reward,frame,epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()

        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = '' if epsilon is None else f", Epsilon : {epsilon:.3f}"
        template = colored(f"Frame {frame} ({len(self.total_rewards)} games) || Reward : {mean_reward:.2f}"
                           f" Speed : {speed:.2f} frame/sec" + epsilon_str,'cyan')
        print(template)

        if epsilon is not None:
            self.writer.add_scalar('epsilon',epsilon,frame)
        self.writer.add_scalar('reward_moving_avg100',mean_reward,frame)
        self.writer.add_scalar('speed',speed,frame)
        self.writer.add_scalar('reward',reward,frame)

        if mean_reward > self.stop_reward:
            print(colored(f"Solved in {len(self.total_rewards)} Games!"),'yellow')
            return True
        return False

def unpack_batch(batch):
    states,actions,rewards,dones,next_states = [], [], [], [], []
    for exp in batch:
        states.append(np.array(exp.state,dtype=np.float32,copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            next_states.append(np.array(exp.state,dtype=np.float32,copy=False))
        else:
            next_states.append(np.array(exp.last_state,dtype=np.float32,copy=False))

    return np.array(states,copy=False),np.array(actions),np.array(rewards,dtype=np.float32)\
        ,np.array(dones,dtype=np.uint8),np.array(next_states,copy=False)

def calc_loss(batch,net,target_net,gamma,device = 'cpu'):
    states,actions,rewards,dones,next_states = unpack_batch(batch)

    states_var = torch.tensor(states).to(device)
    next_states_var = torch.tensor(next_states).to(device)
    actions_var = torch.tensor(actions,dtype=torch.int64).to(device)
    rewards_var = torch.tensor(rewards).to(device)
    dones_mask = torch.tensor(dones,dtype=torch.bool).to(device)

    state_action_values = net(states_var).gather(1,actions_var.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_var).max(1)[0]
    next_state_values[dones_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_var
    return nn.MSELoss()(expected_state_action_values,state_action_values)
