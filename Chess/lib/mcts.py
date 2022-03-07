import math
import collections
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import game
class MCTS:
    def __init__(self,c_punc=1.0,env=None):
        assert env is not None
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
        # waiting dict : Used to boost search efficiency
        self.waiting = collections.defaultdict(list)

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.values_avg.clear()
        self.probs.clear()

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
        state_b = state_np.tobytes()
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

            if cur_state_b == state_b:
                noises = np.random.dirichlet([0.03] * self.action_size)
                probs = [0.75 * prob + 0.25 * noise for prob,noise in zip(probs,noises)]

            score = [q_value + self.c_punc * prob * total_sqrt / (1 + count)
                     for q_value,prob,count in zip(values_avg,probs,counts)]

            illegal_actions = set(self.all_action_list) - set(env_copy.legal_actions)
            for illegal_action in illegal_actions:
                score[illegal_action] = -np.inf

            for waited_action in self.waiting[cur_state_b]:
                score[waited_action] = -1000

            action = int(np.argmax(score))
            actions.append(action)

            cur_state_np, reward, done, info = env_copy.step(action)
            cur_player = 1 - cur_player
            if done:
                if cur_player == game.WHITE:
                    value = reward
                else:
                    value = -reward

        return value, cur_state_np, cur_player, states_b, actions

    def is_leaf(self,state_np):
        return state_np.tobytes() not in self.probs

    def search_batch(self,count,batch_size,state_np,player,net,device='cpu'):
        for i in range(count):
            self.search_minibatch(batch_size,state_np,player,net,device=device)

    def search_minibatch(self,count,state_np,player,net,device='cpu'):
        backup_queue = []
        expand_states_np = []
        expand_players = []
        expand_queue = []
        planned_b = set()

        for i in range(count):
            value, leaf_state_np, leaf_player, states_b, actions = self.find_leaf(state_np,player)
            leaf_state_b = leaf_state_np.tobytes()
            if value is not None:
                backup_queue.append((value,states_b,actions))
            else:
                if leaf_state_b not in planned_b:
                    if actions:
                        self.waiting[states_b[-1]].append(actions[-1])
                    planned_b.add(leaf_state_b)
                    expand_states_np.append(leaf_state_np)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state_b,states_b,actions))


        if expand_queue:
            self.waiting.clear()
            batch_var = game.states_np_list_to_batch(expand_states_np,expand_players,device)

            logits_var, values_var = net(batch_var)
            probs_var = F.softmax(logits_var,dim=1)
            values_np = values_var.data.to('cpu').numpy()[:,0]
            probs_np = probs_var.data.to('cpu').numpy()

            for (leaf_state_b,states_b,actions),value,prob in zip(expand_queue,values_np,probs_np):
                self.visit_count[leaf_state_b] = [0] * self.action_size
                self.value[leaf_state_b] = [0.0] * self.action_size
                self.values_avg[leaf_state_b] = [0.0] * self.action_size
                self.probs[leaf_state_b] = prob
                backup_queue.append((value,states_b,actions))

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

    def update_root_node(self,env,state_np):
        self.env = copy.deepcopy(env)
        self.root_board_np = state_np
        self.root_board_b = self.root_board_np.tobytes()
        self.root_board_str = self.env.render(mode='unicode')









