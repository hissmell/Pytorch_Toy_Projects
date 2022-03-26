import gym
import numpy as np
from c_utils import check_action_type
class Omok(gym.Env):
    '''
    white = -1, black = 1, empty = 0
    observation : np.ndarray (2,board_size,board_size)
    action : np.int64 (0 ~ boardsize*2 - 1)
    '''
    def __init__(self,board_size = 19):
        super(Omok,self).__init__()
        self.action_space = gym.spaces.Discrete(board_size*board_size)
        self.observation_space = gym.spaces.Box(low=-1,high=1,shape=(2,board_size,board_size),dtype=np.float32)

        self.board_size = board_size
        self.board_list = []
        self.board_array = np.zeros([board_size,board_size],dtype=np.float32)

        for _ in range(board_size):
            temp = ['.' for _ in range(self.board_size)]
            self.board_list.append(temp)
        self.turn = -1

        self.legal_moves = []
        for row in range(1,self.board_size+1):
            for col in range(1,self.board_size+1):
                self.legal_moves.append((row, col))
        self.legal_actions = [action for action in range(self.board_size * self.board_size)]

    def current_state_to_observation(self):
        return np.stack((np.ones((self.board_size,self.board_size),dtype=np.float32)*self.turn,self.board_array),axis=0)

    def reset(self):
        self.board_list = []
        self.board_array = np.zeros([self.board_size, self.board_size],np.float32)
        for _ in range(self.board_size):
            temp = ['.' for _ in range(self.board_size)]
            self.board_list.append(temp)
        self.turn = -1

        self.legal_moves = []
        for row in range(1,self.board_size+1):
            for col in range(1,self.board_size+1):
                self.legal_moves.append((row, col))
        self.legal_actions = [action for action in range(self.board_size * self.board_size)]
        return self.current_state_to_observation()

    def step(self,action):
        assert type(action) == int
        observation = None
        reward = None
        done = False
        info = None

        if (action < 0) or (action >= self.board_size*self.board_size):
            raise Exception(f"Input action : {action:d} -> 보드의 범위를 벗어난 action입니다.")

        move = self.decode_action(action)
        x,y = move[0]-1,move[1]-1
        if self.board_array[x][y] != 0:
            raise Exception(f"Input move : ({move[0]:d},{move[1]:d}) -> 이미 그 자리에 돌이 놓여져 있습니다.")

        # 가능한 action에서 이미 두어진 수는 제외합니다.
        move = self.decode_action(action)
        self.legal_moves.remove(move)
        self.legal_actions.remove(action)

        # 금지수를 입력하면 해당 턴에 해당하는 플레이어는 패배합니다.
        action_type = check_action_type(self.board_array,action,self.turn,self.board_size)
        if action_type == 0:
            ''' 금지수가 아니고 승부가 나지 않은 상황입니다. '''
            done = False
            if not self.legal_actions:
                done = True # 무승부가 난 경우입니다.
            reward = 0.

        elif action_type == 1:
            ''' 금지수가 입력된 경우입니다. '''
            done = True
            reward = -1. if self.turn == 1 else -1.

        else:
            ''' 금지수가 아닌 경우로 승리한 경우입니다.'''
            done = True
            reward = 1. if self.turn == -1 else -1.

        # 다음 상태로 이동합니다.
        simbol = 'O' if self.turn == -1 else 'X'
        self.board_list[x][y] = simbol
        self.board_array[x][y] = self.turn
        self.turn = -self.turn

        return self.current_state_to_observation(),reward,done,info

    def render(self,mode = 'unicode'):
        col_coordinate1 = '    '
        n = 1
        for ii in range(2*self.board_size - 1):
            if ii % 2 == 0:
                t = '{:2d}'.format(n)
                n += 1
            else:
                t = ' '
            col_coordinate1 = col_coordinate1 + t
        col_coordinate2 = '    '
        for ii in range(2*self.board_size - 1):
            t = '--' if ii % 2 == 0 else ' '
            col_coordinate2 = col_coordinate2 + t
        print(col_coordinate1)
        print(col_coordinate2)

        for row in range(self.board_size):
            row_string = '{:2d}'.format(row+1) + ' |'
            for col in range(self.board_size):
                row_string = row_string + ' ' + self.board_list[row][col] + ' '
            print(row_string)

    def encode_action(self,move):
        x,y = move
        return (x-1) * self.board_size + (y - 1)

    def decode_action(self,action):
        x = action // self.board_size + 1
        y = action % self.board_size + 1
        return x,y

    def board_encoding(self,board,turn):
        observation = np.zeros(shape = [self.board_size,self.board_size,2],dtype = np.int32)
        player = 1 if turn == 'O' else -1
        observation[:,:,1] = observation[:,:,1] + player
        for raw in range(self.board_size):
            for col in range(self.board_size):
                if board[raw][col] == 'O':
                    stone = 1
                elif board[raw][col] == 'X':
                    stone = -1
                else:
                    stone = 0
                observation[raw][col][0] = stone
        return observation

    def render_observation(self,observation):
        turn = 'O' if observation[0][0][0] == -1 else 'X'
        board_size = observation.shape[-1]

        board = []
        for _ in range(board_size):
            temp = ['.' for _ in range(board_size)]
            board.append(temp)

        for raw in range(board_size):
            for col in range(board_size):
                if observation[1][raw][col] == -1:
                    stone = 'O'
                elif observation[1][raw][col] == 1:
                    stone = 'X'
                else:
                    stone = '.'
                board[raw][col] = stone

        print('Turn :',turn)
        col_coordinate1 = '    '
        n = 1
        for ii in range(2*board_size - 1):
            if ii % 2 == 0:
                t = '{:2d}'.format(n)
                n += 1
            else:
                t = ' '
            col_coordinate1 = col_coordinate1 + t
        col_coordinate2 = '    '
        for ii in range(2*board_size - 1):
            t = '--' if ii % 2 == 0 else ' '
            col_coordinate2 = col_coordinate2 + t
        print(col_coordinate1)
        print(col_coordinate2)

        for row in range(board_size):
            row_string = '{:2d}'.format(row+1) + ' |'
            for col in range(board_size):
                row_string = row_string + ' ' + board[row][col] + ' '
            print(row_string)

    def render(self,mode='unicode'):
        cur_observation = self.current_state_to_observation()

        turn = 'O' if cur_observation[0][0][0] == -1 else 'X'
        board_size = cur_observation.shape[-1]

        board = []
        for _ in range(board_size):
            temp = ['.' for _ in range(board_size)]
            board.append(temp)

        for raw in range(board_size):
            for col in range(board_size):
                if cur_observation[1][raw][col] == -1:
                    stone = 'O'
                elif cur_observation[1][raw][col] == 1:
                    stone = 'X'
                else:
                    stone = '.'
                board[raw][col] = stone

        board_str = 'Turn : ' + turn + '\n'

        col_coordinate1 = '    '
        n = 1
        for ii in range(2 * board_size - 1):
            if ii % 2 == 0:
                t = '{:2d}'.format(n)
                n += 1
            else:
                t = ' '
            col_coordinate1 = col_coordinate1 + t
        col_coordinate2 = '    '
        for ii in range(2 * board_size - 1):
            t = '--' if ii % 2 == 0 else ' '
            col_coordinate2 = col_coordinate2 + t

        board_str = board_str + col_coordinate1 + '\n'
        board_str = board_str + col_coordinate2 + '\n'

        for row in range(board_size):
            row_string = '{:2d}'.format(row+1) + ' |'
            for col in range(board_size):
                row_string = row_string + ' ' + board[row][col] + ' '
            board_str = board_str + row_string + '\n'

        return board_str

if __name__ == '__main__':
    import random
    env = Omok(board_size=13)
    obs = env.reset()
    for _ in range(200):
        print(env.render())
        action = random.choice(env.legal_actions)
        obs,reward,done,_ = env.step(action)

        if done:
            print('Action :', env.decode_action(action))
            print(reward)
            print(env.render())
            break

    import time
    start = time.time()
    steps = 100
    times = 10
    for _ in range(times):
        env.reset()
        for _ in range(steps):
            action = random.choice(env.legal_actions)
            obs, reward, done, _ = env.step(action)

    print(f'Average time per step : {(time.time() - start) / steps * times : .6f} msec')



