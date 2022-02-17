import torch
from torch.nn import Module,Linear,ReLU,Sequential

class DQN(Module):
    def __init__(self,observation_size,action_size):
        super(DQN, self).__init__()
        self.dense_layers = Sequential(
            Linear(observation_size[0], 64),
            ReLU(),
            Linear(64,64),
            ReLU(),
            Linear(64,action_size)
        )

    def forward(self,x):
        return self.dense_layers(x)

class DuelingDQN(Module):
    def __init__(self,observation_size,action_size):
        super(DuelingDQN, self).__init__()
        self.dense_layers = Sequential(
            Linear(observation_size[0], 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
        )

        self.adv_layers = Sequential(
            Linear(64, action_size)
        )

        self.val_layers = Sequential(
            Linear(64, 1)
        )

    def forward(self,x):
        x = self.dense_layers(x)
        adv = self.adv_layers(x)
        val = self.val_layers(x)
        return val + adv - adv.mean()

