import os; import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from termcolor import colored
from tensorboardX import SummaryWriter
from lib import common, models, envs


''' Hyperparameters '''
EXP_NAME = 'NetV01_00' #netV[version_number]_exp_number'
iter_num = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
MAX_EPOCH = 50
LEARNING_RATE = 1e-4
VALUE_WEIGHT = 100

''' Main body '''

path_dict,net,iter_num = common.training_dir_setting(exp_name=EXP_NAME
                                                     ,iter_num=iter_num
                                                     ,device=device)

with open(path_dict['data_dir_path'],'r') as f:
    dataset = json.load(f)

train_dataset = dataset['train']
valid_dataset = dataset['valid']

train_states = np.array(train_dataset['states'],dtype=np.float32)
train_probs = np.array(train_dataset['probs'],dtype=np.float32)
train_rewards = np.array(train_dataset['rewards'],dtype=np.float32)

valid_states = np.array(valid_dataset['states'],dtype=np.float32)
valid_probs = np.array(valid_dataset['probs'],dtype=np.float32)
valid_rewards = np.array(valid_dataset['rewards'],dtype=np.float32)

print('Train state shape :',train_states.shape)
print('Train prob shape :',train_probs.shape)
print('Train reward shape :',train_rewards.shape)
print('Valid state shape :',valid_states.shape)
print('Valid prob shape :',valid_probs.shape)
print('Valid reward shape :',valid_rewards.shape)

writer = SummaryWriter(path_dict['record_dir_path'],comment=EXP_NAME + str(iter_num))
optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)
best_valid_loss = 10000
for epoch in range(MAX_EPOCH):
    #training step
    net.train()

    shuffle_indices = np.random.permutation(train_states.shape[0])
    train_states = train_states[shuffle_indices]
    train_probs = train_probs[shuffle_indices]
    train_rewards = train_rewards[shuffle_indices]

    train_total_loss = 0.0
    train_policy_loss = 0.0
    train_value_loss = 0.0

    upper_limit = 0
    index = 0
    while train_states.shape[0] > upper_limit:
        upper_limit += BATCH_SIZE
        upper_limit = min(upper_limit,train_states.shape[0])
        optimizer.zero_grad()

        batch_states_arr = train_states[index:upper_limit]
        batch_probs_arr = train_probs[index:upper_limit]
        batch_values_arr = train_rewards[index:upper_limit]

        batch_states_var = torch.tensor(batch_states_arr,dtype=torch.float32,device=device)
        batch_probs_var = torch.tensor(batch_probs_arr,dtype=torch.float32,device=device)
        batch_values_var = torch.tensor(batch_values_arr,dtype=torch.float32,device=device)

        pred_logits_var,pred_values_var = net(batch_states_var)
        pred_values_var = pred_values_var.squeeze(-1)

        loss_value_var = F.mse_loss(batch_values_var,pred_values_var,reduction='sum') * VALUE_WEIGHT
        loss_policy_var = -F.log_softmax(pred_logits_var,dim=1) * batch_probs_var
        loss_policy_var = loss_policy_var.sum()

        total_loss_var = loss_value_var + loss_policy_var

        train_total_loss += total_loss_var.to('cpu').data
        train_policy_loss += loss_policy_var.to('cpu').data
        train_value_loss += loss_value_var.to('cpu').data

        total_loss_var.backward()
        optimizer.step()

        index = upper_limit

    writer.add_scalar('Train_total_loss',train_total_loss/train_states.shape[0],epoch)
    writer.add_scalar('Train_policy_loss',train_policy_loss/train_states.shape[0],epoch)
    writer.add_scalar('Train_value_loss',train_value_loss/train_states.shape[0],epoch)

    #validation step
    net.eval()

    shuffle_indices = np.random.permutation(valid_states.shape[0])
    valid_states = valid_states[shuffle_indices]
    valid_probs = valid_probs[shuffle_indices]
    valid_rewards = valid_rewards[shuffle_indices]

    valid_total_loss = 0.0
    valid_policy_loss = 0.0
    valid_value_loss = 0.0

    upper_limit = 0
    index = 0
    while valid_states.shape[0] > upper_limit:
        upper_limit += BATCH_SIZE
        upper_limit = min(upper_limit, valid_states.shape[0])

        batch_states_arr = valid_states[index:upper_limit]
        batch_probs_arr = valid_probs[index:upper_limit]
        batch_values_arr = valid_rewards[index:upper_limit]

        batch_states_var = torch.tensor(batch_states_arr, dtype=torch.float32, device=device)
        batch_probs_var = torch.tensor(batch_probs_arr, dtype=torch.float32, device=device)
        batch_values_var = torch.tensor(batch_values_arr, dtype=torch.float32, device=device)

        pred_logits_var, pred_values_var = net(batch_states_var)
        pred_values_var = pred_values_var.squeeze(-1)

        loss_value_var = F.mse_loss(batch_values_var, pred_values_var, reduction='sum') * VALUE_WEIGHT
        loss_policy_var = -F.log_softmax(pred_logits_var, dim=1) * batch_probs_var
        loss_policy_var = loss_policy_var.sum()

        total_loss_var = loss_value_var + loss_policy_var

        valid_total_loss += total_loss_var.to('cpu').data
        valid_policy_loss += loss_policy_var.to('cpu').data
        valid_value_loss += loss_value_var.to('cpu').data

        index = upper_limit

    writer.add_scalar('Valid_total_loss', valid_total_loss / valid_states.shape[0], epoch)
    writer.add_scalar('Valid_policy_loss', valid_policy_loss / valid_states.shape[0], epoch)
    writer.add_scalar('Valid_value_loss', valid_value_loss / valid_states.shape[0], epoch)

    print(colored('-'*20,'red'))
    print(colored(f'Epoch : {epoch} / {MAX_EPOCH}','red'))
    print(colored(f'Train Loss : {train_total_loss/train_states.shape[0]:.4f} (total) ||'
                  f' {train_policy_loss/train_states.shape[0]:.4f} (policy) ||'
                  f' {train_value_loss/train_states.shape[0]:.4f} (value)\n'
                  f'Valid Loss : {valid_total_loss / valid_states.shape[0]:.4f} (total) ||'
                  f' {valid_policy_loss / valid_states.shape[0]:.4f} (policy) ||'
                  f' {valid_value_loss / valid_states.shape[0]:.4f} (value)','cyan'))
    print()

    if best_valid_loss > (valid_total_loss / valid_states.shape[0]):
        best_valid_loss = valid_total_loss / valid_states.shape[0]
        torch.save(net.state_dict(),path_dict['net_save_path'])
        print(f'saved! with value_total_loss : {best_valid_loss:.6f}')






