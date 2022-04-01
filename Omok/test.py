from lib import envs
from lib import common
from lib import models
from lib import mcts

net1_path = './saves/Exp_01/best_001_00000_performance_0.3333.pth'
net2_path = None


env = envs.Omok(board_size=9)
net1 = common.load_model(net1_path)
net2 = common.load_model(net2_path)
mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]

common.play_game(env,mcts_stores,None,net1,net2,10,160,8,False,'cpu',True,False)



