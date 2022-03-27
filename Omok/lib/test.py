from mcts import MCTS
import numpy as np
BLACK = -1
WHITE = 1
if __name__ == '__main__':
    from envs import Omok

    env = Omok(8)
    obs = env.reset()
    mcts1 = MCTS(env)
    mcts2 = MCTS(env)

    turn = 'O'
    done = False
    while not done:
        print(env.render())
        if turn == 'O':
            mcts1.update_root_node(env,obs)
            mcts1.search_batch(count=128,batch_size=64,state_np=obs,player=BLACK,net=None,device='cpu')
            prob, value = mcts1.get_policy_value(obs,tau=0)
            action = np.random.choice(mcts1.all_action_list,p=prob)
            print("Player 1 Nodes :", len(mcts1.probs))

            N_dict,Q_dict = mcts1.get_root_child_statistics()
            top = min(3,len(env.legal_actions))
            N_list = sorted(list(N_dict.keys()),key = lambda x : N_dict[x],reverse=True)
            Q_list = sorted(list(Q_dict.keys()),key = lambda x : Q_dict[x],reverse=True)
            for i in range(1,top+1):
                print(f'Top {i} Action :',N_list[i-1],
                      f'Visit : {N_dict[N_list[i-1]]} Q_value : {Q_dict[Q_list[i-1]]:.3f}')

        else:
            move = input("Enter the coordinate : ").split(' ')
            x,y = list(map(int,move))
            action = env.encode_action((x,y))

        action = int(action)
        obs,reward,done,_ = env.step(action)
        turn = 'O' if turn == 'X' else 'X'
    print(f'Reward : {reward}')
    print('Action :',env.decode_action(action))
    print(env.render())