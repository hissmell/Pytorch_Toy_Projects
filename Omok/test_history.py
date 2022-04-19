from lib import envs
from lib import common
from lib import models
from lib import mcts

env = envs.Omok(board_size=9)
net = models.Net(env.observation_space.shape,env.action_space.n)
mcts_stores = mcts.MCTS(env)
_, game_steps, game_history = common.play_game(env, mcts_stores, replay_buffer=None
                                              ,net1=net, net2=net
                                              ,steps_before_tau_0=10
                                              ,mcts_searches=100
                                              ,mcts_batch_size=8, device='cuda'
                                              ,render=True,return_history=True)
for state,_,_,_ in game_history:
    print(env.render_observation(state))

common.render_history(env,game_history)