import gym
import gym_chess
import chess
import numpy as np
import random

import mcts

class RespondLegalAndIllegalAction(gym.Wrapper):
    def __init__(self,env=None):
        super(RespondLegalAndIllegalAction,self).__init__(env)

    def step(self,action):
        if action in self.env.legal_actions:
            return self.env.step(action)
        else:
            action_random = random.choice(self.env.legal_actions)
            obs_next,reward,done,info = self.env.step(action_random)
            # if turn == 'White'
            if obs_next[:,:,-7][0][0] == 0:
                reward = -1.
            else:
                reward = 1.
            done = True
            info = 1
            return obs_next,reward,done,info

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=np.min(self.env.observation_space.low),
                                                high=np.max(self.env.observation_space.high),
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

def make_env(env_id):
    assert env_id == 'ChessAlphaZero-v0'
    env = gym.make(env_id)
    env = RespondLegalAndIllegalAction(env)
    env = ImageToPyTorch(env)
    return env


def play_game(env, mcts_stores, replay_buffer, net1, net2, steps_before_tau_0
              , mcts_searches, mcts_batch_size, net1_plays_first=None, device='cpu'):
    action_size = env.action_space.n

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    elif isinstance(mcts_stores, mcts.MCTS()):
        mcts_stores = [mcts_stores, mcts_stores]

    state_np = env.reset()
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0 if net1_plays_first else 1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        mcts_stores[cur_player].search_batch(mcts_searches, mcts_batch_size,
                                             state_np, cur_player, nets[cur_player], device=device)
        probs, _ = mcts_stores[cur_player].get_policy_value(state_np, tau=tau)
        game_history.append((state_np, cur_player, probs))
        action = np.random.choice(action_size, p=probs)

        state_np, reward, done, info = env.step(action)
        if done:
            result = reward
            net1_result = reward if cur_player == 0 else -reward
            break

        cur_player = 1 - cur_player
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_buffer is not None:
        for state_np, cur_player, probs in reversed(game_history):
            replay_buffer.append((state_np, cur_player, probs, result))
            result = -result

    return net1_result, step



if __name__ == '__main__':
    env_id = 'ChessAlphaZero-v0'
    env = make_env(env_id)

    turn = 'White'
    max_len = 100

    obs = env.reset()
    done = False
    reward = 0.0
    print('Turn :', turn)
    print('Before Reward :', reward)
    print(env.render(mode='unicode'))

    # turn = 'Black' if turn == 'White' else 'White'
    # action = random.choice(env.legal_actions)
    # obs, reward, done, info = env.step(action)
    # print('\n----------------------')
    # print('Turn :', turn)
    # print('Before Reward :', reward)
    # print(env.render(mode='unicode'))



    while not done:
        turn = 'Black' if turn == 'White' else 'White'
        action = random.choice(range(1000))
        obs, reward, done, info = env.step(action)
        print('\n----------------------')
        print('Turn :', turn)
        print('Before Reward :', reward)
        print(env.render(mode='unicode'))

    env.close()

