from lib import common, envs, models
import torch
import numpy as np

net_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01\\' \
           'Pytorch_Toy_Projects\\Omok\\NetV01_00\\iter_01\\net_path.pth'
device = 'cuda'
net = common.load_model(net_path,device)
net.eval()
net.to(device)

env = envs.Omok(board_size=9)
top = 3

step = 0
obs = env.reset()
done = False
print(env.render())
while not done:
    move = input('Enter coordinate (x,y) : ').split(',')
    move = tuple(map(int,move))
    obs,reward,done,_ = env.step(env.encode_action(move))
    prob,value = net(torch.tensor(obs.reshape(-1,*obs.shape),dtype=torch.float32).to(device))
    prob_np = prob.squeeze(0).data.to('cpu').numpy()
    prob_np = np.exp(prob_np) / np.sum(np.exp(prob_np))
    value = value.squeeze().data.to('cpu').numpy()
    try:
        for ii in range(1,top+1):
            t_a = np.argmax(prob_np)
            t_m = env.decode_action(t_a)
            print(f'Top Action {ii:d} : {t_m[0]:d}, {t_m[1]:d} (Prob {prob_np[t_a]*100:.2f}%)')
            prob_np[t_a] = -1
    except:
        pass
    print(f'Board Value : {value:.4f}')
    print(env.render())




