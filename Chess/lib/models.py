import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


''' Network '''
class PoolAndInject(nn.Module):
    def __init__(self,input_size):
        super(PoolAndInject, self).__init__()
        self.input_size = input_size
        self.global_avg_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self,x):
        x_ = self.global_avg_pool(x)
        x_ = torch.tile(x_,(1,1,self.input_size[1],self.input_size[2]))
        return torch.cat((x,x_),dim=1)

class FeatureExtrudeBlock(nn.Module):
    def __init__(self,input_size):
        super(FeatureExtrudeBlock, self).__init__()

        self.pool_and_inject = PoolAndInject(input_size)
        pool_and_inject_out_channel = self._get_pool_and_inject_out_channel(input_size)

        self.line1 = nn.Sequential(
            nn.Conv2d(pool_and_inject_out_channel,32,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.line2 = nn.Sequential(
            nn.Conv2d(pool_and_inject_out_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.line3 = nn.Sequential(
            nn.Conv2d(pool_and_inject_out_channel, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )

        self.dim_adjust = nn.Conv2d(32+64+64,64,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.pool_and_inject(x)
        x1 = self.line1(x)
        x2 = self.line2(x)
        x3 = self.line3(x)
        x_out = self.dim_adjust(torch.cat((x1,x2,x3),dim=1))
        return x_out

    def _get_pool_and_inject_out_channel(self,input_size):
        temp = torch.zeros(1,*input_size,dtype=torch.float32)
        temp = self.pool_and_inject(temp)
        return int(temp.size()[1])

class ResidualBlock(nn.Module):
    def __init__(self,input_channel,kernel_size,stride,padding):
        super(ResidualBlock, self).__init__()
        self.line1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(input_channel),
            nn.LeakyReLU(),
            nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(input_channel),
        )

        self.leaky_relu = nn.LeakyReLU()

    def forward(self,x):
        x_ = self.line1(x)
        x = self.leaky_relu(x + x_)
        return x

class PolicyHead(nn.Module):
    def __init__(self,input_shape,action_size):
        super(PolicyHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        conv_out = self._get_conv_out(input_shape)

        self.policy = nn.Linear(conv_out,action_size)

    def _get_conv_out(self,input_shape):
        temp = torch.zeros(1,*input_shape,dtype=torch.float32)
        temp = self.conv(temp)
        return int(np.prod(temp.size()))

    def forward(self,x):
        x = self.conv(x).view(x.size()[0],-1)
        policy = self.policy(x)
        return policy

class ValueHead(nn.Module):
    def __init__(self,input_shape):
        super(ValueHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        conv_out = self._get_conv_out(input_shape)

        self.value = nn.Linear(conv_out, 1)

    def _get_conv_out(self, input_shape):
        temp = torch.zeros(1, *input_shape, dtype=torch.float32)
        temp = self.conv(temp)
        return int(np.prod(temp.size()))

    def forward(self, x):
        x = self.conv(x).view(x.size()[0],-1)
        value = self.value(x)
        return value

class Net(nn.Module):
    def __init__(self,state_size,action_size):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.conv_in = nn.Conv2d(state_size[0],128,kernel_size=1,stride=1,padding=0)
        self.feature_extrude_block = FeatureExtrudeBlock(input_size=(128,*state_size[1:]))
        self.residual_block1 = ResidualBlock(input_channel=64,kernel_size=3,stride=1,padding=1)
        self.residual_block2 = ResidualBlock(input_channel=64,kernel_size=3,stride=1,padding=1)
        self.residual_block3 = ResidualBlock(input_channel=64,kernel_size=3,stride=1,padding=1)
        self.residual_block4 = ResidualBlock(input_channel=64,kernel_size=3,stride=1,padding=1)
        self.residual_block5 = ResidualBlock(input_channel=64,kernel_size=3,stride=1,padding=1)
        self.residual_block6 = ResidualBlock(input_channel=64,kernel_size=3,stride=1,padding=1)

        common_layers_out = self._get_common_layers_out_shape(state_size)
        self.policy_head = PolicyHead(common_layers_out,action_size)
        self.value_head = ValueHead(common_layers_out)

    def _get_common_layers_out_shape(self,input_shape):
        temp = torch.zeros(1,*input_shape,dtype=torch.float32)
        temp = self.conv_in(temp)
        temp = self.feature_extrude_block(temp)
        temp = self.residual_block1(temp)
        temp = self.residual_block2(temp)
        temp = self.residual_block3(temp)
        temp = self.residual_block4(temp)
        temp = self.residual_block5(temp)
        temp = self.residual_block6(temp)
        return temp.size()[1:]

    def forward(self,x):
        x = self.conv_in(x)
        x = self.feature_extrude_block(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


if __name__ == '__main__':
    import game
    env = game.make_env('ChessAlphaZero-v0')
    obs = env.reset()

    test_input = torch.tensor(obs,dtype=torch.float32).unsqueeze(dim=0)
    net = Net(env.observation_space.shape,env.action_space.n)
    p,v = net(test_input)
    print(p.size())
    print(v.size())




