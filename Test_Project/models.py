import torch
from torch.nn import Module,Linear,ReLU,Conv2d,Flatten,Softmax,Dropout2d

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=32,stride=(1,1),padding=1,kernel_size=(3,3))
        self.dropout1 = Dropout2d(p=0.5)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(in_channels=32,out_channels=32,stride=(1,1),padding=1,kernel_size=(3,3))
        self.dropout2 = Dropout2d(p=0.5)
        self.relu2 = ReLU()
        self.conv3 = Conv2d(in_channels=32,out_channels=32,stride=(1,1),padding=1,kernel_size=(3,3))
        self.dropout3 = Dropout2d(p=0.5)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(in_channels=32,out_channels=16,stride=(1,1),padding=1,kernel_size=(3,3))
        self.dropout4 = Dropout2d(p=0.5)
        self.relu4 = ReLU()
        self.flatten = Flatten(start_dim=1)
        self.dense = Linear(in_features=16384,out_features=2)
        self.softmax = Softmax(dim=1)

    def forward(self,x):
        x = self.relu1(self.dropout1(self.conv1(x)))
        x = self.relu2(self.dropout2(self.conv2(x)))
        x = self.relu3(self.dropout3(self.conv3(x)))
        x = self.relu4(self.dropout4(self.conv4(x)))
        x = self.dense(self.flatten(x))
        x = self.softmax(x)
        return x


