import torch
import numpy as np
from torch.nn.functional import one_hot
from torch.distributions import Categorical

class ActorCriticAgent:
    def __init__(self,model,optimizer,gamma,action_space_size,state_space_size,device = 'cpu'):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.action = None
        self.state = None
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

        self.replay_buffer = []

    def _observation_to_state(self,observation):
        '''
        :param observation: np.ndarray (4,) 형식
        :return: state: 형식 = torch.array (1,4) // dtype = torch.float32
        '''
        return torch.unsqueeze(torch.from_numpy(observation).type(torch.FloatTensor),dim=0)

    def get_best_action_from_observation(self,observation):
        with torch.no_grad():
            state = self._observation_to_state(observation).to(self.device)
            policy, _ = self.model(state)
            action = torch.argmax(torch.squeeze(policy))
        return action.to('cpu').numpy()

    def get_best_action(self,state):
        with torch.no_grad():
            policy, _ = self.model(state)
            action = torch.argmax(torch.squeeze(policy))
        return action.to('cpu').numpy()


    def get_action_from_observation(self,observation):
        with torch.no_grad():
            state = self._observation_to_state(observation).to(self.device)
            policy, _ = self.model(state)
            sampler = Categorical(probs=torch.squeeze(policy))
            action = sampler.sample()
        return action.to('cpu').numpy()

    def get_action(self,state):
        with torch.no_grad():
            policy, _ = self.model(state)
            sampler = Categorical(probs=torch.squeeze(policy))
            action = sampler.sample()
        return action.to('cpu').numpy()

    def fit(self):
        self.optimizer.zero_grad()
        # Prepare Training Data
        data_num = len(self.replay_buffer)
        states = [replay[0] for replay in self.replay_buffer]
        action_indexes = [replay[1] for replay in self.replay_buffer]
        rewards = [replay[2] for replay in self.replay_buffer]
        dones = [replay[3] for replay in self.replay_buffer]
        next_states = [replay[4] for replay in self.replay_buffer]

        states = torch.from_numpy(np.stack(states,axis = 0).astype(np.float32)).to(self.device)
        action_indexes = torch.from_numpy(np.array(action_indexes)).type(torch.LongTensor).to(self.device)
        actions = one_hot(action_indexes,self.action_space_size).type(torch.FloatTensor).to(self.device)
        rewards = torch.unsqueeze(torch.from_numpy(np.array(rewards).astype(np.float32)).to(self.device),dim=1)
        dones = torch.unsqueeze(torch.from_numpy(np.array(dones).astype(np.float32)).to(self.device),dim=1)
        next_states = torch.from_numpy(np.stack(next_states,axis=0).astype(np.float32)).to(self.device)

        # train
        policy,value = self.model(states)
        _,next_value = self.model(next_states)
        next_value = next_value * (1.0 - dones)
        delta = rewards + self.gamma * next_value - value
        delta_detached = delta.detach()

        value_loss = torch.sum(torch.square(delta))
        policy_loss = torch.sum(-torch.multiply(torch.log(torch.sum(policy*actions,dim=1,keepdim=True)),delta_detached))
        loss = value_loss + policy_loss
        loss = torch.sum(loss) / data_num

        # Backpropagation & Update model
        loss.backward()
        self.optimizer.step()

        # Clear the cache in GPU
        torch.cuda.empty_cache()

        return loss.detach().item(),policy_loss.detach().item()/data_num,value_loss.detach().item()/data_num

    def append_replay_buffer(self,observation,action_index,reward,done,next_state):
        '''
        :param observation: np.ndarray (4,)의 행렬
        :param action_index: np.int32
        :param reward: np.float32
        :param done: np.float32
        :param next_state: np.ndarray (4,) 의 행렬
        :return:
        '''
        self.replay_buffer.append(tuple([observation,action_index,reward,done,next_state]))

    def reset_replay_buffer(self):
        self.replay_buffer = []

    def to(self,device = 'cpu'):
        self.device = device
        self.model.to(device)

