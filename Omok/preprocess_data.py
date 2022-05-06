import os; import json
import numpy as np

from lib import common

''' Hyperparameters '''
EXP_NAME = 'NetV00_00' #netV[version_number]_exp_number'
iter_num = None # if you put None as input, last_iter_num is returned
TRAIN_RATIO = 0.9

''' Main body '''

path_dict,iter_num = common.preprocessing_dir_setting(exp_name=EXP_NAME,iter_num=iter_num)
game_list = os.listdir(path_dict['data_dir_path'])
max_data_num = len(game_list)

sampling_num = iter_num + 1
gamma = 1 - ((1/sampling_num) ** 2)
states_list = []
probs_list = []
rewards_list = []
for data_num in range(max_data_num):
    game_path = os.path.join(path_dict['data_dir_path'],f'game_{data_num}.json')
    with open(game_path,'r') as f:
        game_data = json.load(f)
    game_states_list = game_data['states']
    game_probs_list = game_data['probs']
    game_rewards_list = game_data['results']

    game_length = len(game_states_list)
    for ri in range(-sampling_num,0):
        states_list.append(game_states_list[ri])
        probs_list.append(game_probs_list[ri])
        discounted_reward = ((-gamma) ** (abs(ri)-1)) * game_rewards_list[ri]
        rewards_list.append(discounted_reward)

    middle_index = np.random.permutation(game_length-sampling_num)
    middle_index = middle_index[:sampling_num]
    for mi in list(middle_index):
        states_list.append(game_states_list[mi])
        probs_list.append(game_probs_list[mi])
        discounted_reward = ((-gamma) ** (game_length-1-mi)) * game_rewards_list[mi]
        rewards_list.append(discounted_reward)

    if (data_num+1) % 1000 == 0:
        print(f"Progress : {data_num} / {max_data_num}")

train_data_num = int(len(states_list) * TRAIN_RATIO)
valid_data_num = max_data_num - train_data_num

train_states_list = states_list[:train_data_num]
train_probs_list = probs_list[:train_data_num]
train_rewards_list = rewards_list[:train_data_num]
train_dataset = {'states' : train_states_list
                 ,'probs' : train_probs_list
                 ,'rewards' : train_rewards_list}

valid_states_list = states_list[train_data_num:]
valid_probs_list = probs_list[train_data_num:]
valid_rewards_list = rewards_list[train_data_num:]
valid_dataset = {'states' : valid_states_list
                 ,'probs' : valid_probs_list
                 ,'rewards' : valid_rewards_list}

dataset = {'train':train_dataset,'valid':valid_dataset}
with open(os.path.join(path_dict['iter_dir_path'],'dataset.json'),'w') as f:
    json.dump(dataset,f)

game_data = {'states':states_list,'probs':probs_list,'results':rewards_list}
