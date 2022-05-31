from lib import common, envs, models, mcts
import torch
import numpy as np
from termcolor import colored

device = 'cpu'
env = envs.Omok(board_size=9)
result = None
player_turn = int(input('Choose your color [-1 or 1] : '))
cur_player = -1
state = env.reset()
net_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01' \
           '\\Pytorch_Toy_Projects\\Omok\\NetV01_00\\iter_01\\net_path.pth'
net = models.NetV2()
net.load_state_dict(torch.load(net_path,map_location=device))
net.eval()
tree = mcts.MCTS(env)

while result is None:
    print(env.render())

    if not cur_player == player_turn:
        tree.update_root_node(env, state, net, device=device)
        tree.search_batch(200, 8, state, cur_player, net, device=device)
        probs, _ = tree.get_policy_value(state, tau=0.08)
        action = int(np.random.choice(tree.all_action_list, p=probs))
    else:
        action_str_list = input("Input Action (x,y): ").split(',')
        action_int_list = map(int,action_str_list)
        action = env.encode_action(action_int_list)

    if cur_player == -1:
        if player_turn == 1:
            print(colored(f"Turn : Computer (O turn) Nodes {len(tree.probs):}", 'blue'))
    else:
        if player_turn == -1:
            print(colored(f"Turn : Computer (X turn) Nodes {len(tree.probs):}", 'blue'))

    if not cur_player == player_turn:
        N_dict, Q_dict = tree.get_root_child_statistics()
        top = min(3, len(env.legal_actions))
        N_list = sorted(list(N_dict.keys()), key=lambda x: N_dict[x], reverse=True)
        for i in range(1, top + 1):
            print(colored(
                f'Top {i} Action : ({N_list[i - 1][0]:d},{N_list[i - 1][1]:d})'
                f' Visit : {N_dict[N_list[i - 1]]} Q_value : {Q_dict[N_list[i - 1]]:.3f}'
                f' Prob : {probs[env.encode_action(N_list[i - 1])] * 100:.2f} %', 'cyan'))
        move = env.decode_action(action)
        print(colored(f"Action taken : ({move[0]:d},{move[1]:d})"
                      f" Visit : {N_dict[move]} Q_value : {Q_dict[move]:.3f}"
                      f" Prob : {probs[env.encode_action(move)] * 100:.2f} %", 'red'))

    state, reward, done, _ = env.step(action)
    if done:
        print(env.render())
    cur_player = -cur_player
