import torch
from torch.nn import Module,Linear,ReLU

class TestModel(Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.dense = Linear(10,10)
        self.relu = ReLU()

    def forward(self,inputs):
        inputs = self.dense(inputs)
        inputs = self.relu(inputs)
        return inputs

