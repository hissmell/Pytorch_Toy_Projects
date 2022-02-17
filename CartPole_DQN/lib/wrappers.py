import gym
import numpy as np

class UnsqueezeObservation(gym.ObservationWrapper):
    def __init__(self,env):
        super(UnsqueezeObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=self.env.observation_space.low.reshape(1,-1),
                                                high=self.env.observation_space.high.reshape(1,-1),
                                                shape=(1,self.env.observation_space.shape[0]))

    def observation(self, obs):
        return obs.reshape(1,-1)

    def reset(self):
        return self.env.reset().reshape(1,-1)

class RewardScaler(gym.RewardWrapper):
    def __init__(self,env):
        super(RewardScaler, self).__init__(env)

    def reward(self, reward):
        reward = 0.1 if reward >= 1 else -1
        return reward

def cartpole_env_make(version):
    assert type(version) == str
    env_name = 'CartPole-' + version

    env = gym.make(env_name)
    env = RewardScaler(env)
    return env


if __name__ == '__main__':
    version = 'v1'
    env = cartpole_env_make(version)
    start_obs = env.reset()
    step_obs = env.step(1)
    print(env.observation_space)
    print(env.action_space)
    print(start_obs)
    print(step_obs)

