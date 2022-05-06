from lib import common, envs, models
import torch
import torch.nn as nn
import numpy as np

net_path = './saves/Exp_01/best_001_00100_performance_0.9500.pth'
device = 'cpu'
net = common.load_model(load_path=net_path,device=device)
print(net)

output_sequence = {}
def hook_fn(module,input,output):
    output_sequence[module] = output

def get_all_layer(net):
    for name,layer in net._modules.items():
        if isinstance(layer,nn.Sequential):
            get_all_layer(layer)
        else:
            layer.register_forward_hook(hook_fn)

get_all_layer(net)

out = net(torch.randn(1,2,9,9).to(device))

for key,value in output_sequence.items():
    print(key)
    value_arr = value.to('cpu').data.numpy()
    print('mean :',value_arr.mean())
    print('var :',value_arr.var())