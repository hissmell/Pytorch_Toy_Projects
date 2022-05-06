import math
import collections
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

BLACK = -1
WHITE = 1

class MCTS:
    def __init__(self,env,c_punc=1.0):
        self.action_size = env.action_space.n
        self.env = copy.deepcopy(env)
        self.root_board_np = env.reset()
        self.root_board_b = self.root_board_np.tobytes()
        self.root_board_str = self.env.render(mode='unicode')
        self.c_punc = c_punc


        self.all_action_list = [act for act in range(self.action_size)]
        # N[s] -> [action count list]
        self.visit_count = {}
        # total value
        self.value = {}
        # Q[s] -> [value avg list], computed by N[s,a] / self.value[s,a]
        self.values_avg = {}
        # P[s] -> [probability list]
        self.probs = {}
        # wating -> key = state_b, value = waiting actions(list)
        self.waiting = defaultdict(list)

    def __len__(self):
        return len(self.probs)

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.values_avg.clear()
        self.probs.clear()
        self.env.reset()

    def find_leaf(self, state_np, player):
        '''
        :param state_np:
        :param player:
        (1) value : None if leaf node, or else equals to the game outcome for the player at leaf node
        (2) leaf_state_np : nd_array bytes of the leaf node
        (3) player : player at the leaf node
        (4) states_b : list of states (in bytes) traversed
        (5) actions : list of actions (int 64) taken
        :return:
        '''
        states_b = []
        actions = []
        cur_state_np = state_np
        cur_player = player
        value = None
        env_copy = copy.deepcopy(self.env)

        while not self.is_leaf(cur_state_np):
            cur_state_b = cur_state_np.tobytes()
            states_b.append(cur_state_b)
            counts = self.visit_count[cur_state_b]
            total_sqrt = math.sqrt(sum(counts))
            probs = self.probs[cur_state_b]
            values_avg = self.values_avg[cur_state_b]

            # if cur_state_b == state_b:
            #     noises = np.random.dirichlet([0.03] * self.action_size)
            #     probs = [0.75 * prob + 0.25 * noise for prob,noise in zip(probs,noises)]

            score = [q_value + self.c_punc * prob * total_sqrt / (1 + count)
                     for q_value,prob,count in zip(values_avg,probs,counts)]

            illegal_actions = set(self.all_action_list) - set(env_copy.legal_actions)
            for illegal_action in illegal_actions:
                score[illegal_action] = -np.inf

            for waiting_action in self.waiting[cur_state_b]:
                score[waiting_action] = -100

            action = int(np.argmax(score))
            actions.append(action)

            cur_state_np, reward, done, info = env_copy.step(action)
            cur_player = -cur_player
            if done:
                if cur_player == BLACK:
                    value = reward
                else:
                    value = -reward

        return value, cur_state_np, cur_player, states_b, actions

    def is_leaf(self,state_np):
        return state_np.tobytes() not in self.probs

    def search_batch(self,count,batch_size,state_np,player,net,device='cpu'):
        for i in range(count):
            self.search_minibatch(batch_size,state_np,player,net,device=device)

    def search_minibatch(self,count,state_np,player,net=None,device='cpu'):
        backup_queue = []
        expand_states_np = []
        expand_players = []
        expand_queue = []

        for i in range(count):
            value, leaf_state_np, leaf_player, states_b, actions = self.find_leaf(state_np,player)
            leaf_state_b = leaf_state_np.tobytes()
            if value is not None:
                backup_queue.append((value,states_b,actions))
            else:
                self.waiting[states_b[-1]].append(actions[-1])
                expand_states_np.append(leaf_state_np)
                expand_players.append(leaf_player)
                expand_queue.append((leaf_state_b,states_b,actions))

        if expand_queue:
            if not net == None:
                batch_var = torch.tensor(np.array(expand_states_np), dtype=torch.float32).to(device)
                logits_var, values_var = net(batch_var)
                probs_var = F.softmax(logits_var,dim=1)
                values_np = values_var.data.to('cpu').numpy()[:,0] # 흑 플레이어 입장에서 value입니다.
                probs_np = probs_var.data.to('cpu').numpy()
            else:
                N = np.array(expand_states_np).shape[0]
                probs_np = np.ones([N,self.action_size],dtype=np.float32) / self.action_size
                values_np = np.ones([N],dtype=np.float32) / self.action_size

            # 흑 플레이어 입장에서 예측된 value를 해당 노드 플레이어 입장에서 value로 조정합니다.
            for ii in range(len(expand_players)):
                values_np[ii] = values_np[ii] if expand_players[ii] == BLACK else -values_np[ii]

            for (leaf_state_b,states_b,actions),value,prob in zip(expand_queue,values_np,probs_np):
                self.visit_count[leaf_state_b] = [0] * self.action_size
                self.value[leaf_state_b] = [0.0] * self.action_size
                self.values_avg[leaf_state_b] = np.random.uniform(low=-0.01,high=0.01,size=self.action_size)
                self.probs[leaf_state_b] = prob
                backup_queue.append((value,states_b,actions))
        self.waiting.clear()

        for value,states_b,actions in backup_queue:
            '''
            leaf state is not stored in states_b.
            therefore value is supposed to opponent's value.
            so we have to convert it to reverse!
            '''
            cur_value = -value
            for state_b, action in zip(states_b[::-1],actions[::-1]):
                self.visit_count[state_b][action] += 1
                self.value[state_b][action] += cur_value
                self.values_avg[state_b][action] = self.value[state_b][action] / self.visit_count[state_b][action]
                cur_value = -cur_value



    def get_policy_value(self,state_np,tau=1):
        '''
        Extract policy and action-value by the state(nd.array)
        :param state_np:
        :param tau:
        :return: (probs,values)
        '''
        state_b = state_np.tobytes()
        counts = self.visit_count[state_b]
        if tau == 0:
            probs = [0.0] * self.action_size
            probs[np.argmax(counts)] = 1.0
        else:
            counts_np = np.power(np.array(counts), (1 / tau))
            probs = list(counts_np / np.sum(counts_np))

        values = self.values_avg[state_b]
        return probs,values

    def get_root_child_statistics(self):
        legal_actions = self.env.legal_actions
        N_dict = {}
        Q_dict = {}
        for legal_action in legal_actions:
            N_dict[self.env.decode_action(legal_action)] = self.visit_count[self.root_board_b][legal_action]
            Q_dict[self.env.decode_action(legal_action)] = self.values_avg[self.root_board_b][legal_action]
        return N_dict,Q_dict


    def update_root_node(self,env,state_np,net=None,device='cpu'):
        self.env = copy.deepcopy(env)
        self.root_board_np = state_np
        self.root_board_b = self.root_board_np.tobytes()
        self.root_board_str = self.env.render(mode='unicode')
        if self.is_leaf(self.root_board_np):
            if net is not None:
                batch_var = torch.tensor(np.array([state_np]),dtype=torch.float32).to(device)
                logits_var,values_var = net(batch_var)
                probs_var = F.softmax(logits_var, dim=1)
                values_np = values_var.data.to('cpu').numpy()[:, 0]
                probs_np = probs_var.data.to('cpu').numpy()
            else:
                N = 1
                probs_np = np.ones([N, self.action_size], dtype=np.float32) / self.action_size
                values_np = np.ones([N], dtype=np.float32) / self.action_size

            self.visit_count[self.root_board_b] = [0] * self.action_size
            self.value[self.root_board_b] = [0.0] * self.action_size
            self.values_avg[self.root_board_b] = np.random.uniform(low=-0.01,high=0.01,size=self.action_size)
            self.probs[self.root_board_b] = probs_np[0]



if __name__ == '__main__':
    from envs import Omok

    env = Omok(9)
    obs = env.reset()
    mcts1 = MCTS(env)
    mcts2 = MCTS(env)

    turn = 'O'
    done = False
    while not done:
        print(env.render())
        if turn == 'O':
            mcts1.update_root_node(env,obs)
            mcts1.search_batch(count=160,batch_size=8,state_np=obs,player=BLACK,net=None,device='cpu')
            prob, value = mcts1.get_policy_value(obs,tau=0.1)
            action = np.random.choice(mcts1.all_action_list,p=prob)
            print("Player 1 Nodes :", len(mcts1.probs))

            N_dict,Q_dict = mcts1.get_root_child_statistics()
            top = min(3,len(env.legal_actions))
            N_list = sorted(list(N_dict.keys()),key = lambda x : N_dict[x],reverse=True)
            Q_list = sorted(list(Q_dict.keys()),key = lambda x : Q_dict[x],reverse=True)
            for i in range(1,top+1):
                print(f'Top {i} Action :',N_list[i-1],
                      f'Visit : {N_dict[N_list[i-1]]} Q_value : {Q_dict[Q_list[i-1]]:.3f}')

        else:
            mcts2.update_root_node(env,obs)
            mcts2.search_batch(count=160,batch_size=8,state_np=obs,player=WHITE,net=None,device='cpu')
            prob, value = mcts2.get_policy_value(obs,tau=0.1)
            action = np.random.choice(mcts2.all_action_list,p=prob)
            print("Player 2 Nodes :", len(mcts2.probs))
            N_dict, Q_dict = mcts2.get_root_child_statistics()
            top = min(3, len(env.legal_actions))
            N_list = sorted(list(N_dict.keys()), key=lambda x: N_dict[x], reverse=True)
            Q_list = sorted(list(Q_dict.keys()), key=lambda x: Q_dict[x], reverse=True)
            for i in range(1, top + 1):
                print(f'Top {i} Action :', N_list[i - 1],
                      f'Visit : {N_dict[N_list[i - 1]]} Q_value : {Q_dict[Q_list[i - 1]]:.3f}')

        action = int(action)
        obs,reward,done,_ = env.step(action)
        turn = 'O' if turn == 'X' else 'X'
    print(f'Reward : {reward}')
    print('Action :',env.decode_action(action))
    print(env.render())


