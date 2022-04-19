from lib import envs
from lib import common
from lib import models
from lib import mcts

net1_path = './saves/Exp_01/best_001_00100_performance_0.6000.pth'
net2_path = None


env = envs.Omok(board_size=9)
net1 = common.load_model(net1_path)
net2 = common.load_model(net2_path)
mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]
result = 0
for _ in range(10):
    r,_,_ = common.play_game(env,mcts_stores,None,net1,net2,10,160,8,False,'cuda',True,False)
    mcts_stores = [mcts.MCTS(env), mcts.MCTS(env)]
    result += r

print(result)


