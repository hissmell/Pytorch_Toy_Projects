import torch
from torch.nn import Module,Linear,ReLU,Conv2d,MaxPool2d,Flatten,ConvTranspose2d

class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),dtype=torch.float32)
        self.maxpool1 = MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.relu1 = ReLU()
        self.conv2 = Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),dtype=torch.float32)
        self.maxpool2 = MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.relu2 = ReLU()
        self.conv3 = Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),dtype=torch.float32)
        self.relu3 = ReLU()
        self.flatten = Flatten(start_dim=1)
        self.mean = Linear(3136,3,dtype=torch.float32)
        self.log_variance = Linear(3136,3,dtype=torch.float32)

    def forward(self,x):
        x = self.relu1(self.maxpool1(self.conv1(x)))
        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        means = self.mean(x)
        variances = torch.exp(self.log_variance(x))
        return means,variances

class Sampler(Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self,means,variances,device='cuda'):
        batch_size = means.size()[0]
        z = variances * torch.normal(torch.zeros(size=(batch_size,3),dtype=torch.float32).to(device),torch.ones(size=(batch_size,3),dtype=torch.float32).to(device)) + means
        return z

class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense = Linear(3,3136,dtype=torch.float32)
        self.relu1 = ReLU()
        self.deconv1 = ConvTranspose2d(64,64,kernel_size=(4,4),stride=(2,2),padding=(1,1),dtype=torch.float32)
        self.relu2 = ReLU()
        self.deconv2 = ConvTranspose2d(64, 1, kernel_size=(4,4), stride=(2,2), padding=(1,1), dtype=torch.float32)

    def forward(self,x):
        x = self.relu1(self.dense(x))
        x = x.view(-1,64,7,7)
        x = self.relu2(self.deconv1(x))
        x = self.deconv2(x)
        return x

class VAE(Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.sampler = Sampler()
        self.decoder = Decoder()

    def forward(self,x,device='cuda'):
        means,variances = self.encoder(x)
        z = self.sampler(means,variances,device)
        output = self.decoder(z)
        return output,means,variances



if __name__ =='__main__':
    import numpy as np
    print(torch.Tensor([[0.,0.,0.],[0.,0.,0.]]))
    print(torch.normal(mean=torch.Tensor([[0.,0.,0.],[0.,0.,0.]]),std=torch.Tensor([[1.,1.,1.],[1.,1.,1.]])))
    test_input = torch.from_numpy(np.random.randn(64,1,28,28).astype(np.float32))
    model = VAE()
    test_ouput,test_means,test_variances = model(test_input)
    print(test_ouput.size())
    print(test_ouput.dtype)
    print(test_means.size())
    print(test_variances.size())
    print(model)
