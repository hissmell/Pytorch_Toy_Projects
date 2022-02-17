from termcolor import colored
import time
import numpy as np

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
        self.total_rewards = []

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0

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

