import torch
from torch.nn import Module,Linear,ReLU,Softmax,Tanh

class PolicyHead(Module):
    def __init__(self,input_feature_size,action_space_size):
        super(PolicyHead, self).__init__()
        self.action_space_size = action_space_size
        self.dense1 = Linear(input_feature_size,64,dtype=torch.float32)
        self.relu = ReLU()
        self.dense2 = Linear(64,action_space_size,dtype=torch.float32)
        self.softmax = Softmax(dim=1)

    def forward(self,x):
        x = self.relu(self.dense1(x))
        policy = self.softmax(self.dense2(x))
        return policy

class ValueHead(Module):
    def __init__(self,input_feature_size):
        super(ValueHead, self).__init__()
        self.dense1 = Linear(input_feature_size,64,dtype=torch.float32)
        self.relu = ReLU()
        self.dense2 = Linear(64,1,dtype=torch.float32)
        self.tanh = Tanh()

    def forward(self,x):
        x = self.relu(self.dense1(x))
        value = self.tanh(self.dense2(x))
        return value

class ActorCriticModel(Module):
    def __init__(self,input_feature_size,latent_space_size,action_space_size):
        super(ActorCriticModel, self).__init__()
        self.action_space_size = action_space_size
        self.latent_space_size = latent_space_size
        self.input_feature_size = input_feature_size

        # 메인 계산 레이어
        self.dense1 = Linear(input_feature_size,64,dtype=torch.float32)
        self.relu1 = ReLU()
        self.dense2 = Linear(64,64,dtype=torch.float32)
        self.relu2 = ReLU()
        self.dense3 = Linear(64,latent_space_size,dtype=torch.float32)
        self.relu3 = ReLU()

        # 출력 레이어
        self.policy_head = PolicyHead(input_feature_size=latent_space_size,action_space_size=action_space_size)
        self.value_head = ValueHead(input_feature_size=latent_space_size)

    def forward(self,x):
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy,value

if __name__ == '__main__':
    import numpy as np
    latent_space_size = 64
    action_space_size = 2

    test_model = ActorCriticModel(input_feature_size=4
                                  ,latent_space_size=latent_space_size
                                  ,action_space_size=action_space_size)

    test_input = torch.from_numpy(np.random.randn(2,4).astype(np.float32))
    test_output_p,test_output_v = test_model(test_input)
    print(test_output_p)
    print(test_output_p.size())
    print(test_output_p.dtype)
    print(test_output_v)
    print(test_output_v.size())
    print(test_output_v.dtype)

