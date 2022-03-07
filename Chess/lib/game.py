import gym
import gym_chess
import chess
import numpy as np
import random
import torch
import collections
import chess
import copy
import re

from termcolor import colored
from lib import mcts
from lib import models

''' GAME SETTING '''
WHITE = 0
BLACK = 1

''' WRAPPERS '''

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

''' For Rendering '''
def edited_render(render_str):
    return re.sub('⭘',chr(9675),render_str)

class Render:
    def __init__(self):
        self.observe_prev = None

    def revise_render(self,board_str,current_turn):
        observe_cur = edited_render(board_str)
        observe_out = copy.deepcopy(observe_cur)
        if self.observe_prev is not None:
            c = 0
            for idx in range(len(observe_cur)):
                if self.observe_prev[idx] != observe_cur[idx]:
                    if current_turn == WHITE:
                        observe_out = observe_out[:idx+(9*c)] + colored(observe_out[idx+(9*c)],'red')\
                                      + observe_out[idx+(9*c)+1:]
                    else:
                        observe_out = observe_out[:idx+(9*c)] + colored(observe_out[idx+(9*c)],'cyan')\
                                      + observe_out[idx+(9*c)+1:]
                    c += 1
        self.observe_prev = observe_cur
        return observe_out





def states_np_list_to_batch(expand_states_np,expand_players,device):
    return torch.tensor(np.array(expand_states_np),dtype=torch.float32).to(device)

def play_game(env, net1, net2, steps_before_tau_0, mcts_searches, mcts_batch_size
              ,replay_buffer=None,mcts_stores=None, net1_plays_first=None, device='cpu',render=False,render_worker=None):
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net1, models.Net)
    assert isinstance(net2, models.Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    action_size = env.action_space.n

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(env=env), mcts.MCTS(env=env)]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state_np = env.reset()
    nets = [net1.to(device), net2.to(device)]
    net1_color = WHITE
    if net1_plays_first is None:
        if np.random.rand() > 0.5:
            nets = [net2,net1]
            net1_color = BLACK

    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    cur_player = WHITE
    while result is None:
        if render:
            turn = 'White' if cur_player == WHITE else 'Black'
            board_str = render_worker.revise_render(env.render(mode='unicode'),current_turn=cur_player)
            print('--------------------------------------------')
            print(f'Step : {step} ({turn} turn)')
            print(board_str)
            print()
        mcts_stores[cur_player].update_root_node(env, state_np)
        mcts_stores[cur_player].search_batch(mcts_searches, mcts_batch_size,
                                             state_np, cur_player, nets[cur_player], device=device)
        print("Player 1 Nodes :",len(mcts_stores[net1_color].probs))
        print("Player 2 Nodes :",len(mcts_stores[1-net1_color].probs))
        probs, value = mcts_stores[cur_player].get_policy_value(state_np, tau=tau)


        game_history.append((state_np, cur_player, probs))
        action = np.random.choice(action_size, p=probs)
        print('Action :',action,f'(visited {mcts_stores[cur_player].visit_count[state_np.tobytes()][action]})')

        state_np, reward, done, info = env.step(action)
        if done:
            if render:
                turn = 'White' if cur_player == WHITE else 'Black'
                board_str = render_worker.revise_render(env.render(mode='unicode'), current_turn=cur_player)
                print('--------------------------------------------')
                print(f'Step : {step} ({turn} turn)')
                print(board_str)
                print()

            result = reward
            net1_result = reward if cur_player == net1_color else -reward
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

