import torch
import torch.nn as nn
import torch.nn.functional as F
import common
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

        self.deconv = nn.ConvTranspose2d(64,1,kernel_size=4,stride=4,padding=1)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        reward_conv_out = self._get_conv_out((self.input_channels,) + state_size[1:],self.reward_conv)

        self.reward_dense = nn.Sequential(
            nn.Linear(reward_conv_out,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def _get_conv_out(self,input_shape,conv_sequential):
        temp = torch.zeros(1,*input_shape,dtype=torch.float32)
        temp = conv_sequential(temp)
        return int(np.prod(temp.size()))

    def forward(self,images,actions):
        batch_size = common.BATCH_SIZE
        action_planes_var = torch.zeros(batch_size,self.action_size,*self.input_shape[1:],dtype=torch.float32).to(actions.device)
        action_planes_var[range(batch_size),self.action_size] = 1.0
        combined_input_var = torch.cat((images,action_planes_var),dim=1)
        combined_out1_var = self.conv1(combined_input_var)
        combined_out2_var = self.conv2(combined_out1_var)
        combined_out2_var += combined_out1_var

        images_out_var = self.deconv(combined_out2_var)
        reward_conv_var = self.reward_conv(combined_out2_var).view(batch_size,-1)
        reward_out_var = self.reward_dense(reward_conv_var)
        return images_out_var, reward_out_var



