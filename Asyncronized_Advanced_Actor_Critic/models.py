import torch
from torch.nn import Module,Linear,ReLU,Softmax,Tanh

class PolicyNetwork(Module):
    def __init__(self,input_dim = 4, action_size = 2):
        super(PolicyNetwork, self).__init__()
        self.p_dense1 = Linear(input_dim,64)
        self.p_relu1 = ReLU()
        self.p_dense2 = Linear(64,64)
        self.p_relu2 = ReLU()
        self.p_dense3 = Linear(64,action_size)
        self.p_softmax = Softmax(dim=1)

    def forward(self,x):
        x = self.p_relu1(self.p_dense1(x))
        x = self.p_relu2(self.p_dense2(x))
        x = self.p_softmax(self.p_dense3(x))
        return x

class ValueNetwork(Module):
    def __init__(self, input_dim=4):
        super(ValueNetwork, self).__init__()
        self.v_dense1 = Linear(input_dim, 64)
        self.v_relu1 = ReLU()
        self.v_dense2 = Linear(64, 64)
        self.v_relu2 = ReLU()
        self.v_dense3 = Linear(64, 1)
        self.v_tanh = Tanh()

    def forward(self, x):
        x = self.v_relu1(self.v_dense1(x))
        x = self.v_relu2(self.v_dense2(x))
        x = self.v_tanh(self.v_dense3(x))
        return x

class SeparateActorCriticNetwork(Module):
    def __init__(self,state_size=4,action_size=2):
        super(SeparateActorCriticNetwork, self).__init__()
        self.policy_network = PolicyNetwork(state_size,action_size)
        self.value_network = ValueNetwork(state_size)

    def forward(self,x):
        policy = self.policy_network(x)
        value = self.value_network(x)
        return policy,value

class MergeActorCriticNetwork(Module):
    def __init__(self,state_size,action_size,latent_size):
        super(MergeActorCriticNetwork, self).__init__()
        pass

    def forward(self,x):
        pass
