import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import common
import numpy as np

''' Hyperparameters '''
EM_OUT_SHAPE = (1,) + common.IMAGE_SHAPE[1:]

''' Models '''
class EnvironmentModel(nn.Module):
    def __init__(self,state_size,action_size):
        super(EnvironmentModel, self).__init__()

        self.input_shape = state_size
        self.action_size = action_size

        self.input_channels = self.input_shape[0] + self.action_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels,64,kernel_size=4,stride=4),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU()
        )

        self.deconv = nn.ConvTranspose2d(64,1,kernel_size=4,stride=4)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        reward_conv_out = self._get_reward_conv_out((self.input_channels,) + state_size[1:])

        self.reward_dense = nn.Sequential(
            nn.Linear(reward_conv_out,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def _get_reward_conv_out(self,input_shape):
        temp = torch.zeros(1,*input_shape,dtype=torch.float32)
        temp = self.conv1(temp)
        temp = self.conv2(temp)
        temp = self.reward_conv(temp)
        return int(np.prod(temp.size()))

    def forward(self,images,actions):
        batch_size = common.BATCH_SIZE
        action_planes_var = torch.zeros(batch_size,self.action_size,*self.input_shape[1:],dtype=torch.float32).to(actions.device)
        action_planes_var[range(batch_size),actions] = 1.0
        combined_input_var = torch.cat((images,action_planes_var),dim=1)
        combined_out1_var = self.conv1(combined_input_var)
        combined_out2_var = self.conv2(combined_out1_var)
        combined_out2_var = combined_out2_var + combined_out1_var

        images_out_var = self.deconv(combined_out2_var)
        reward_conv_var = self.reward_conv(combined_out2_var).view(batch_size,-1)
        reward_out_var = self.reward_dense(reward_conv_var)
        return images_out_var, reward_out_var


class A2C(nn.Module):
    def __init__(self,state_size,action_size):
        super(A2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(state_size)

        self.base = nn.Sequential(
            nn.Linear(conv_out_size,512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512,action_size)
        self.value = nn.Linear(512,1)

    def forward(self,x):
        x = self.conv(x).view(x.size()[0],-1)
        x = self.base(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def _get_conv_out(self,input_size):
        temp_var = torch.zeros(1,*input_size,dtype=torch.float32)
        temp_var = self.conv(temp_var)
        return int(np.prod(temp_var.size()))

if __name__ =='__main__':
    test_input = torch.zeros(common.BATCH_SIZE,2,84,84,dtype=torch.float32)
    test_action = torch.zeros(common.BATCH_SIZE,dtype=torch.int64)

    test_model = EnvironmentModel(state_size=(2,84,84),action_size=4)

    test_images_out_var,test_rewards_out_var = test_model(test_input,test_action)
    print(test_images_out_var.size())
    print(test_rewards_out_var.size())
    test = test_images_out_var.sum()
    test.backwrad()
