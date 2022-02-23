import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvironmentModel(nn.Module):
    def __init__(self,state_size,action_size):
        super(EnvironmentModel, self).__init__()

        self.input_shape = state_size
        self.action_size = action_size

        self.input_channels = self.input_shape[0] + self.action_size

        self.conv =