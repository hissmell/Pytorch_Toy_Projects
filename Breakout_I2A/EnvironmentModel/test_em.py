from lib import common
from lib import models

import ptan
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

def compare_imagination_to_real(net_em,net_act,env,imagin_depth=1,prev_obs=None,device='cuda'):
    if prev_obs is None:
        prev_obs = env.reset()

    assert type(prev_obs) == np.ndarray
    real_prev_obs_var = torch.tensor(np.array([prev_obs]), dtype=torch.float32).to(device)
    imagin_prev_obs_var = torch.tensor(np.array([prev_obs]), dtype=torch.float32).to(device)

    net_act.to(device)
    net_em.to(device)

    real_image_list = []
    imagin_image_list = []
    real_reward_list = []
    imagin_reward_list = []
    for _ in range(imagin_depth):
        real_action_logit_var = net_act(real_prev_obs_var)[0]
        real_action_prob_np = F.softmax(real_action_logit_var,dim=1).squeeze(dim=0).to('cpu').data.numpy()
        real_action = np.random.choice(env.action_space.n,p=real_action_prob_np)

        imgain_action_logit_var = net_act(imagin_prev_obs_var)[0]
        imgain_action_prob_np = F.softmax(imgain_action_logit_var,dim=1).squeeze(dim=0).to('cpu').data.numpy()
        imgain_action = np.random.choice(env.action_space.n,p=imgain_action_prob_np)
        imgain_action_var = torch.tensor(np.array([imgain_action]),dtype=torch.int64).to(device)

        real_obs_np,real_reward_np,done,_ = env.step(real_action)
        real_obs_np = real_obs_np[-1] # 프레임을 스택하였기 때문에 가장 마지막 채널에 있는 이미지가 새로 생성된 이미지임!

        imagin_obs_diff_var,imagin_reward_var = net_em(imagin_prev_obs_var,imgain_action_var)
        imagin_obs_var = imagin_prev_obs_var
        imagin_obs_var[-1] = imagin_obs_var[-1] + imagin_obs_diff_var # net_em이 예측하는 것이 (새로 생성되는 이미지 - 이전 이미지) 이기 때문이다.
        imagin_reward_np = float(imagin_reward_var.to('cpu').data.numpy())
        imagin_obs_np = imagin_obs_var.squeeze().to('cpu').data.numpy()
        imagin_obs_np = imagin_obs_np[-1]
        imagin_obs_np = np.clip(imagin_obs_np, 0.0, 1.0)


        assert real_obs_np.shape == (84,84)
        assert imagin_obs_np.shape == (84, 84)
        real_obs_var = torch.cat((real_prev_obs_var,torch.tensor(real_obs_np.reshape(1,1,84,84)).to(device)),dim=1)
        real_obs_var = real_obs_var[0,-2:,:,:].unsqueeze(dim=0)
        real_prev_obs_var = real_obs_var
        imagin_prev_obs_var = imagin_obs_var

        real_obs_np = real_obs_np.squeeze() * 255
        imagin_obs_np = imagin_obs_np.squeeze() * 255

        real_obs_np = real_obs_np.astype(np.uint8)
        imagin_obs_np = imagin_obs_np.astype(np.uint8)

        real_image_list.append(real_obs_np)
        imagin_image_list.append(imagin_obs_np)
        real_reward_list.append(real_reward_np)
        imagin_reward_list.append(imagin_reward_np)

        if done:
            break

    imagin_depth = len(real_image_list)
    plt.rcParams['figure.figsize'] = (5.0 * imagin_depth,10)
    rows = imagin_depth
    columns = 2

    for r in range(rows):
        real_obs_image = Image.fromarray(real_image_list[r], mode='L')
        imagin_obs_image = Image.fromarray(imagin_image_list[r], mode='L')

        real_title = f"Real (depth {r}) Reward : {real_reward_list[r]:.4f}"
        plt.subplot(rows,columns,r*columns + 1)
        plt.title(real_title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(real_obs_image)

        imagin_title = f"Imagined (depth {r}) Reward : {imagin_reward_list[r]:.4f}"
        plt.subplot(rows,columns,r*columns+2)
        plt.title(imagin_title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagin_obs_image)
    plt.show()

def imagination(net_em,env,current_obs_np,device='cuda'):
    current_obs_var = torch.tensor(current_obs_np.reshape(1,*current_obs_np.shape),dtype=torch.float32).to(device)
    imagin_obs_list = []
    action_str_list = []
    imagin_rew_list = []
    plt.rcParams['figure.figsize'] = (7,7.0 * env.action_space.n)
    for action in range(env.action_space.n):
        action_str = env.unwrapped.get_action_meanings()[action]
        imagin_image_diff_var, imagin_reward_var = net_em(current_obs_var,torch.tensor([action],dtype=torch.int64).to(device))
        imagin_image_diff_np = imagin_image_diff_var.squeeze().data.to('cpu').numpy()
        imagin_reward_np = imagin_reward_var.squeeze().data.to('cpu').numpy()

        imagin_image_np = current_obs_np[-1] + imagin_image_diff_np
        imagin_image_np = np.clip(imagin_image_np,0.0,1.0)
        imagin_image_np = imagin_image_np*255
        imagin_image_np = imagin_image_np.astype(np.uint8)
        imagin_obs_list.append(imagin_image_np)
        action_str_list.append(action_str)
        imagin_rew_list.append(imagin_reward_np)

    rows = env.action_space.n
    columns = 2

    for r in range(rows):
        if r == 0:
            current_obs_np = current_obs_np[-1] * 255
            current_obs_np = current_obs_np.astype(np.uint8)
            real_obs_image = Image.fromarray(current_obs_np, mode='L')
            real_title = f"Root Image"
            plt.subplot(rows,columns,r*columns + 1)
            plt.title(real_title)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(real_obs_image)

        imagin_obs_image = Image.fromarray(imagin_obs_list[r], mode='L')
        imagin_title = f"Act : {action_str_list[r]} | R: {imagin_rew_list[r]:.4f}"
        plt.subplot(rows,columns,r*columns+2)
        plt.title(imagin_title)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagin_obs_image)
    plt.show()





if __name__ == '__main__':
    device = 'cpu'
    env = common.make_env(test=False)[0]
    act_model_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01\\Pytorch_Toy_Projects\\Breakout_I2A' \
                     '\\Baseline_A2C\\save\\Exp_02\\Exp_02-frame=1906389-score=25.310000-test=337.20.pth'
    em_model_path = 'C:\\Users\\82102\\PycharmProjects\\ToyProject01\\Pytorch_Toy_Projects\\Breakout_I2A\\' \
                    'EnvironmentModel\\save\\Exp_05_Ver3_100_1\\frame=5567423_loss=0.033674.pth'
    net_act = models.A2C(env.observation_space.shape,env.action_space.n).to(device)
    net_em = models.EnvironmentModelVer3(env.observation_space.shape, env.action_space.n).to(device)
    net_act.load_state_dict(torch.load(act_model_path,map_location=torch.device(device)))
    net_em.load_state_dict(torch.load(em_model_path,map_location=torch.device(device)))

    prev_obs = env.reset()
    for _ in range(100):
        action_logit_var = net_act(torch.tensor(np.array([prev_obs]),dtype=torch.float32).to(device))[0]
        action_prob_np = F.softmax(action_logit_var,dim=1).squeeze(dim=0).to('cpu').data.numpy()
        action = np.random.choice(env.action_space.n,p=action_prob_np)
        obs,reward,_,_ = env.step(action)
        if reward == 1:
            print('!')
            break
        prev_obs = obs

    imagination(net_em,env,current_obs_np=prev_obs,device=device)
    compare_imagination_to_real(net_em,net_act,env,imagin_depth=3,prev_obs=prev_obs,device=device)





