import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiConvResBlock(nn.Module):
    def __init__(self,in_channel,n1,n2,n3,n4):
        super(MultiConvResBlock, self).__init__()

        # equivalent to kernel size 7
        self.conv_line1 = nn.Sequential(
            nn.Conv2d(in_channel,n1,kernel_size=(1,1),stride=(1,1),padding=(0,0)),
            nn.Conv2d(n1,n1,kernel_size=(7,7),stride=(1,1),padding=(3,3)),
        )
        # equivalent to kernel size 5
        self.conv_line2 = nn.Sequential(
            nn.Conv2d(in_channel, n2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(n2, n2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )
        # equivalent to kernel size 3
        self.conv_line3 = nn.Sequential(
            nn.Conv2d(in_channel, n3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(n3, n3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        # equivalent to kernel size 1
        self.conv_line4 = nn.Sequential(
            nn.Conv2d(in_channel, n4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

        # Combine a range of conv_outputs
        self.conv_combine = nn.Sequential(
            nn.Conv2d(n1+n2+n3+n4,in_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )

    def forward(self,x):
        x1 = self.conv_line1(x)
        x2 = self.conv_line2(x)
        x3 = self.conv_line3(x)
        x4 = self.conv_line4(x)
        x_comb = torch.cat([x1,x2,x3,x4],dim=1)
        x_out = self.conv_combine(x_comb)
        return x + x_out


class Net(nn.Module):
    def __init__(self,state_size,action_size):
        super(Net, self).__init__()

        self.feature_extruder = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=(1,5),stride=(1,1),padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=(5,1),stride=(1,1),padding=(2,0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_res_blocks = nn.Sequential(
            MultiConvResBlock(64, 32, 32, 32, 32),
            MultiConvResBlock(64, 32, 32, 32, 32),
            MultiConvResBlock(64, 32, 32, 32, 32),
        )

        conv_out = self.get_conv_out(state_size)
        self.policy = nn.Sequential(
            nn.Linear(conv_out,512),
            nn.ReLU(),
            nn.Linear(512,action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out,512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Tanh()
        )

    def get_conv_out(self,state_size):
        temp = torch.zeros(1,*state_size,dtype=torch.float32)
        temp = self.feature_extruder(temp)
        temp = self.conv_res_blocks(temp)
        return int(np.prod(temp.size()))

    def forward(self,x):
        x = self.feature_extruder(x)
        x = self.conv_res_blocks(x)
        x = x.view(x.size()[0],-1)
        policy = self.policy(x)
        value = self.value(x)
        return policy,value

class ConvolutionalBlock(nn.Module):
    def __init__(self,inchannel_num,filter_num = 256):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(inchannel_num,filter_num,kernel_size=3,stride=1,padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=filter_num)
        self.relu = nn.ReLU()

    def forward(self,input):
        x = self.conv(input)
        x = self.batch_norm(x)
        output = self.relu(x)
        return output

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,filter_num=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,filter_num,kernel_size=3,stride=1,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(filter_num)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filter_num,filter_num,kernel_size=3,stride=1,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(filter_num)
        self.relu2 = nn.ReLU()

    def forward(self,input):
        x = self.conv1(input)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.add(x,input)
        x = self.relu2(x)
        return x

class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()
        self.convolutional_block = ConvolutionalBlock(2)
        self.residual_tower = nn.Sequential(*[
            ResidualBlock(256) for _ in range(19)
        ])

        #policy head
        self.p_conv = nn.Conv2d(256,2,kernel_size=1,stride=1,padding=0)
        self.p_batch_norm = nn.BatchNorm2d(2)
        self.p_relu = nn.ReLU()
        policy_conv_out = self._get_policy_conv_out()
        self.p_dense = nn.Linear(policy_conv_out,81)

        #value head
        self.v_conv = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        self.v_batch_norm1 = nn.BatchNorm2d(1)
        self.v_relu1 = nn.ReLU()
        value_conv_out = self._get_value_conv_out()
        self.v_dense1 = nn.Linear(value_conv_out,256)
        self.v_relu2 = nn.ReLU()
        self.v_dense2 = nn.Linear(256,1)
        self.v_tanh = nn.Tanh()

    def forward(self,input):
        x = self.convolutional_block(input)
        x = self.residual_tower(x)

        p = self.p_conv(x)
        p = self.p_batch_norm(p)
        p = self.p_relu(p)
        p = self.p_dense(p.view(p.size()[0],-1))

        v = self.v_conv(x)
        v = self.v_batch_norm1(v)
        v = self.v_relu1(v)
        v = self.v_dense1(v.view(v.size()[0],-1))
        v = self.v_relu2(v)
        v = self.v_dense2(v)
        v = self.v_tanh(v)
        return p,v

    def _get_policy_conv_out(self):
        temp = torch.zeros(1,2,9,9)
        temp = self.convolutional_block(temp)
        temp = self.residual_tower(temp)
        temp = self.p_conv(temp)
        temp = self.p_batch_norm(temp)
        temp = self.p_relu(temp)
        return int(np.prod(temp.size()))

    def _get_value_conv_out(self):
        temp = torch.zeros(1,2,9,9)
        temp = self.convolutional_block(temp)
        temp = self.residual_tower(temp)
        temp = self.v_conv(temp)
        temp = self.v_batch_norm1(temp)
        temp = self.v_relu1(temp)
        return int(np.prod(temp.size()))






if __name__ == '__main__':
    from envs import Omok
    env = Omok(board_size=9)
    net = Net(env.observation_space.shape,env.action_space.n)
    net.to('cuda')
    obs = env.reset()
    obs_var = torch.tensor(np.array([obs]),dtype=torch.float32).to('cuda')

    p,v = net(obs_var)
    print(p.size(),v.size())
    print(p)
    print(v)
    import time
    ts = time.time()
    for _ in range(500):
        _ = net(obs_var)
    print(f'Time : {time.time() - ts : .6f} sec')
