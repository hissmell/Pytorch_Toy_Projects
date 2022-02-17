import torch
from torch.nn import Module,Linear,ReLU,Sequential

class DQN(Module):
    def __init__(self,observation_size,action_size):
        super(DQN, self).__init__()
        self.dense_layers = Sequential(
            Linear(observation_size[1], 64),
            ReLU(),
            Linear(64,64),
            ReLU(),
            Linear(64,action_size)
        )

    def forward(self,x):
        return self.dense_layers(x)


