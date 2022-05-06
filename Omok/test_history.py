from lib import envs
from lib import common
from lib import models
from lib import mcts
import torch
import os
import json
import numpy as np

EXP_NAME = 'NetV00_00' #netV[version_number]_exp_number'
iter_num = None # if you put None as input, last_iter_num is returned
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_dict,net,iter_num = common.gathering_dir_setting(exp_name=EXP_NAME
                                                          ,iter_num=iter_num
                                                          ,device=device)

env = envs.Omok(board_size=9)
game_num = 7
with open(os.path.join(path_dict['iter_dir_path'],f'dataset.json'),'r') as f:
    game_data = json.load(f)['valid']

print(len(game_data['states']))
print(len(game_data['probs']))
print(len(game_data['rewards']))

common.render_game_data(game_data)