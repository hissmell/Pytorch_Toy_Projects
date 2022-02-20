import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

if __name__ == '__main__':
    model = A2C((2,84,84),4)
    temp_var = torch.zeros(1, *(2,84,84), dtype=torch.float32)
    logits,values = model(temp_var)
    print(model)
