import torch
import numpy as np
from lib import mcts
from lib import envs
from termcolor import colored
import multiprocessing as mp


def play_game(env,mcts_stores,replay_buffer,net1,net2
              ,steps_before_tau_0,mcts_searches,mcts_batch_size
              ,net1_plays_first=False,device='cpu',render=False
              ,return_history=False):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]
    elif isinstance(mcts_stores,mcts.MCTS):
        mcts_stores = [mcts_stores,mcts_stores]

    state = env.reset()
    nets = [net1.to(device),net2.to(device)]

    if not net1_plays_first:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0
    if cur_player == 0:
        net1_color = -1
    else:
        net1_color = 1

    step = 0
    tau = 0.08 if steps_before_tau_0 >= step else 1
    game_history = []

    result = None
    net1_result = None
    turn = -1 # (-1) represents Black turn! (1 does White turn)
    while result is None:
        if render:
            print(env.render())

        mcts_stores[cur_player].update_root_node(env,state,nets[cur_player],device=device)
        mcts_stores[cur_player].search_batch(mcts_searches,mcts_batch_size
                                             ,state,turn,nets[cur_player]
                                             ,device=device)
        probs,_ = mcts_stores[cur_player].get_policy_value(state,tau=tau)
        game_history.append((state,cur_player,probs))
        action = int(np.random.choice(mcts_stores[0].all_action_list, p=probs))

        if render:
            if turn == -1:
                if net1_color == -1:
                    print(colored(f"Turn : Net 1 (O turn) Nodes {len(mcts_stores[cur_player].probs):}",'blue'))
                else:
                    print(colored(f"Turn : Net 2 (O turn) Nodes {len(mcts_stores[cur_player].probs):}", 'blue'))
            else:
                if net1_color == -1:
                    print(colored(f"Turn : Net 2 (X turn) Nodes {len(mcts_stores[cur_player].probs):}",'blue'))
                else:
                    print(colored(f"Turn : Net 1 (X turn) Nodes {len(mcts_stores[cur_player].probs):}", 'blue'))
            N_dict, Q_dict = mcts_stores[cur_player].get_root_child_statistics()
            top = min(3, len(env.legal_actions))
            N_list = sorted(list(N_dict.keys()), key=lambda x: N_dict[x], reverse=True)
            for i in range(1, top + 1):
                print(colored(
                    f'Top {i} Action : ({N_list[i - 1][0]:d},{N_list[i - 1][1]:d})'
                    f' Visit : {N_dict[N_list[i - 1]]} Q_value : {Q_dict[N_list[i - 1]]:.3f}'
                    f' Prob : {probs[env.encode_action(N_list[i - 1])]*100:.2f} %','cyan'))
            move = env.decode_action(action)
            print(colored(f"Action taken : ({move[0]:d},{move[1]:d})"
                          f" Visit : {N_dict[move]} Q_value : {Q_dict[move]:.3f}"
                          f" Prob : {probs[env.encode_action(move)]*100:.2f} %",'red'))


        state,reward,done,_ = env.step(action)
        if done:
            if render:
                print(env.render())
            result = reward
            if net1_color == -1:
                net1_result = reward
            else:
                net1_result = -reward

        cur_player = 1 - cur_player
        turn = -turn

        step += 1
        if step >= steps_before_tau_0:
            tau = 0.08

    if replay_buffer is not None or return_history:
        if return_history:
            h = []
        for state, cur_player, probs in reversed(game_history):
            if replay_buffer is not None:
                replay_buffer.append((state, cur_player, probs, result))
            if return_history:
                h.append((state, cur_player, probs, result))

            result = -result

    return net1_result, step, h

def evaluate_network(env,net1,net2,rounds,search_num=200,batch_size=10,device='cpu',render=False):
    net1_score = 0
    mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]
    for round in range(rounds):
        result,_,_ = play_game(env,mcts_stores,None,net1,net2,0
                             ,search_num,batch_size,False,device
                             ,render,return_history=False)
        net1_score += result

    return net1_score / rounds




if __name__ == '__main__':
    from models import Net
    from mcts import MCTS
    from envs import Omok
    env = Omok(board_size=9)
    mcts_stores = MCTS(env)
    net1 = Net(env.observation_space.shape,env.action_space.n)
    net2 = Net(env.observation_space.shape,env.action_space.n)
    device = 'cuda'
    # render = True
    # net1_result,step = play_game(env,mcts_stores,None,net1,net2,0,mcts_searches=100,mcts_batch_size=10,
    #                         net1_plays_first=False,device=device,render=render)
    #
    # print(net1_result,step)

    print(evaluate_network(env,net1,net2,20,search_num=5,batch_size=5,device=device,render=False))