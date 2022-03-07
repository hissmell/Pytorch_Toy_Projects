from lib import game
from lib import models

import gym
import gym_chess

env_id = 'ChessAlphaZero-v0'
env = game.make_env(env_id)
net1 = models.Net(env.observation_space.shape,env.action_space.n)
net2 = models.Net(env.observation_space.shape,env.action_space.n)
device = 'cpu'
render_worker = game.Render()
net1_result, step = game.play_game(env,net1=net1,net2=net2,steps_before_tau_0=10,mcts_batch_size=64,mcts_searches=5
                                   ,render=True,render_worker=render_worker,device=device)

