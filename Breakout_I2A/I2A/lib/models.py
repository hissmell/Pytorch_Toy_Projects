import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import ptan

''' Hyper-parameters '''

ROLLOUT_HIDDEN = 256
EM_OUT_SHAPE = (1,84,84)

''' Models '''

class RolloutEncoder(nn.Module):
    def __init__(self,input_shape,hidden_shape=ROLLOUT_HIDDEN):
        super(RolloutEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.rnn = nn.LSTM(input_size=conv_out_size+1,hidden_size=hidden_shape,batch_first=False)

    def _get_conv_out(self,input_shape):
        temp = torch.zeros(1,*input_shape,dtype=torch.float32)
        temp = self.conv(temp)
        return int(np.prod(temp.size()))

    def forward(self,obs_var,reward_var):
        n_time = obs_var.size()[0]
        n_batch = obs_var.size()[1]
        n_items = n_time * n_batch

        obs_flat_var = obs_var.view(n_items,*obs_var.size()[2:])
        conv_out = self.conv(obs_flat_var)
        conv_out = conv_out.view(n_time,n_batch,-1)
        rnn_in = torch.cat((conv_out,reward_var),dim=2)
        _,(rnn_hid,_) = self.rnn(rnn_in)
        return rnn_hid.view(-1)


class I2A(nn.Module):
    def __init__(self,state_size,action_size,net_em,net_act,rollout_steps):
        super(I2A, self).__init__()

        self.action_size = action_size
        self.rollout_steps = rollout_steps

        ''' Layers '''
        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(state_size)
        dense_input = conv_out_size + ROLLOUT_HIDDEN * action_size

        self.dense = nn.Sequential(
            nn.Linear(dense_input,512),
            nn.ReLU()
        )

        self.policy = nn.Linear(512,action_size)
        self.value = nn.Linear(512,1)

        ''' Rest '''

        self.encoder = RolloutEncoder(EM_OUT_SHAPE)
        self.action_selector = ptan.actions.ProbabilityActionSelector()

        object.__setattr__(self,"net_em",net_em)
        object.__setattr__(self,"net_act",net_act)



    def _get_conv_out(self,input_shape):
        temp = torch.zeros(1,*input_shape,dtype=torch.float32)
        temp = self.conv(temp)
        return int(np.prod(temp.size()))

    def forward(self,x):
        encode_rollout = self.rollout_batch(x)
        conv_out = self.conv(x).view(x.size()[0],-1)
        dense_in = torch.cat((conv_out,encode_rollout),dim=1)
        dense_out = self.dense(dense_in)
        return self.policy(dense_out), self.value(dense_out)

    def rollout_batch(self,batch):
        batch_size = batch.size()[0]
        batch_rest = batch.size()[1:]
        if batch_size == 1:
            obs_batch_var = batch.expand(batch_size*self.action_size,*batch_rest)
        else:
            obs_batch_var = batch.unsqueeze(1)
            obs_batch_var = obs_batch_var.expand(batch_size,self.action_size,*batch_rest)
            obs_batch_var = obs_batch_var.contiguous().view(-1,batch_rest)

        actions = np.tile(np.arange(0,self.action_size,dtype=np.int64),batch_size)
        step_obs,step_reward = [], []

        for step_idx in range(self.rollout_steps):
            actions_var = torch.tensor(actions).to(batch.device)
            obs_next_var, reward_var = self.net_em(obs_batch_var,actions_var)

            step_obs.append(obs_next_var.detach())
            step_reward.append(reward_var.detach())

            if step_idx == self.rollout_steps - 1:
                break

            cur_plane_var = obs_batch_var[:,1:2]
            new_plane_var = cur_plane_var + obs_next_var
            obs_batch_var = torch.cat((cur_plane_var,new_plane_var),dim=1)

            logits_var,_ = self.net_act(obs_batch_var)
            probs_bar = F.softmax(logits_var,dim=1)
            probs = probs_bar.data.to('cpu').numpy()
            actions = self.action_selector(probs)

        steps_obs_var = torch.stack(step_obs)
        steps_rewards_var = torch.stack(step_reward)
        flat_encoded_var = self.encoder(steps_obs_var,steps_rewards_var)
        return flat_encoded_var.view(batch_size,-1)


