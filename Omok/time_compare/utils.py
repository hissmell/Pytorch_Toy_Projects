import numpy as np
import copy
import gym
from c_check_action_type import check_action_type as c_func

def check_action_type(board_array,action,turn,board_size):
    check_x = action // board_size
    check_y = action % board_size
    ''' 5가 완성되었는지 체크합니다. 그와 동시에 4가 몇개 있는지도 체크합니다. '''
    check_color = turn
    opponent_color = -turn
    max_strike = 0
    count_4 = 0
    # 가로 체크
    y = check_y + 1
    strike = 1
    while 1:
        if y >= board_size or y < 0:
            break
        if board_array[check_x][y] == opponent_color or board_array[check_x][y] == 0:
            break
        if board_array[check_x][y] == turn:
            strike += 1
        y += 1

    y = check_y - 1
    while 1:
        if y >= board_size or y < 0:
            break
        if board_array[check_x][y] == opponent_color or board_array[check_x][y] == 0:
            break
        if board_array[check_x][y] == turn:
            strike += 1
        y -= 1

    if strike == 4:
        count_4 += 1

    if max_strike < strike:
        max_strike = strike

    # 세로 체크
    x = check_x + 1
    strike = 1
    while 1:
        if x >= board_size or x < 0:
            break
        if board_array[x][check_y] == opponent_color or board_array[x][check_y] == 0:
            break
        if board_array[x][check_y] == turn:
            strike += 1
        x += 1

    x = check_x - 1
    while 1:
        if x >= board_size or x < 0:
            break
        if board_array[x][check_y] == opponent_color or board_array[x][check_y] == 0:
            break
        if board_array[x][check_y] == turn:
            strike += 1
        x -= 1

    if strike == 4:
        count_4 += 1

    if max_strike < strike:
        max_strike = strike

    # 좌하향 대각 체크
    x = check_x + 1
    y = check_y - 1
    strike = 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            break
        if board_array[x][y] == opponent_color or board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x += 1
        y -= 1

    x = check_x - 1
    y = check_y + 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            break
        if board_array[x][y] == opponent_color or board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x -= 1
        y += 1

    if strike == 4:
        count_4 += 1

    if max_strike < strike:
        max_strike = strike

    # 우하향 대각 체크
    x = check_x - 1
    y = check_y - 1
    strike = 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            break
        if board_array[x][y] == opponent_color or board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x -= 1
        y -= 1

    x = check_x + 1
    y = check_y + 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            break
        if board_array[x][y] == opponent_color or board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x += 1
        y += 1

    if strike == 4:
        count_4 += 1

    if max_strike < strike:
        max_strike = strike

    if max_strike == 5:
        return 2 # 5가 완성되었음을 의미합니다.

    if turn == 1:
        return 0 # 백으로 두는 경우에는 금지수가 존재하지 않습니다.

    if max_strike > 5:
        return 1 # 장목이 완성되어서 패배하였음을 의미합니다.

    if count_4 >= 2:
        return 1 # 44 자리이므로 패배하였음을 의미합니다.

    ''' 33이 완성되었는지 체크합니다. '''
    count_3 = 0
    # 가로 방향 33 체크
    y = check_y + 1
    empty = 2
    find_stones = 1
    HARD_BOUND = 0
    WEAK_BOUND_L = 0
    AROUND_STONE = 1
    while 1:
        if y >= board_size or y < 0:
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[check_x][y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[check_x][y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[check_x][y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        y += 1

    if find_stones == 1:
        empty = 2
    else:
        empty = 1

    y = check_y - 1
    WEAK_BOUND_R = 0
    AROUND_STONE = 1
    while 1:
        if y >= board_size or y < 0:
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[check_x][y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[check_x][y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[check_x][y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        y -= 1

    if find_stones == 3 and not HARD_BOUND:
        if not (WEAK_BOUND_L and WEAK_BOUND_R):
            count_3 += 1

    # 세로 방향 33 체크
    x = check_x + 1
    empty = 2
    find_stones = 1
    HARD_BOUND = 0
    WEAK_BOUND_L = 0
    AROUND_STONE = 1
    while 1:
        if x >= board_size or x < 0:
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][check_y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][check_y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[x][check_y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        x += 1

    if find_stones == 1:
        empty = 2
    else:
        empty = 1

    x = check_x - 1
    WEAK_BOUND_R = 0
    AROUND_STONE = 1
    while 1:
        if x >= board_size or x < 0:
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][check_y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][check_y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[x][check_y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        x -= 1

    if find_stones == 3 and not HARD_BOUND:
        if not (WEAK_BOUND_L and WEAK_BOUND_R):
            count_3 += 1

    # 좌하향 대각 33 체크
    x = check_x + 1
    y = check_y - 1
    empty = 2
    find_stones = 1
    HARD_BOUND = 0
    WEAK_BOUND_L = 0
    AROUND_STONE = 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[x][y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        x += 1
        y -= 1

    if find_stones == 1:
        empty = 2
    else:
        empty = 1

    x = check_x - 1
    y = check_y + 1
    WEAK_BOUND_R = 0
    AROUND_STONE = 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[x][y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        x -= 1
        y += 1

    if find_stones == 3 and not HARD_BOUND:
        if not (WEAK_BOUND_L and WEAK_BOUND_R):
            count_3 += 1

    # 우하향 대각 33 체크
    x = check_x + 1
    y = check_y + 1
    empty = 2
    find_stones = 1
    HARD_BOUND = 0
    WEAK_BOUND_L = 0
    AROUND_STONE = 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_L = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[x][y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        x += 1
        y += 1

    if find_stones == 1:
        empty = 2
    else:
        empty = 1

    x = check_x - 1
    y = check_y - 1
    WEAK_BOUND_R = 0
    AROUND_STONE = 1
    while 1:
        if (x >= board_size or x < 0) or (y >= board_size or y < 0):
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == opponent_color:
            if not AROUND_STONE:
                WEAK_BOUND_R = 1
            else:
                HARD_BOUND = 1
            break
        if board_array[x][y] == turn:
            find_stones += 1
            AROUND_STONE = 1
        elif board_array[x][y] == 0:
            empty -= 1
            AROUND_STONE = 0
        if empty <= 0:
            break
        x -= 1
        y -= 1

    if find_stones == 3 and not HARD_BOUND:
        if not (WEAK_BOUND_L and WEAK_BOUND_R):
            count_3 += 1
    if count_3 >= 2:
        return 1 # 33이 완성되어 금지수로 판단합니다.

    return 0





class Omok(gym.Env):
    '''
    white = -1, black = 1, empty = 0
    observation : np.ndarray (2,board_size,board_size)
    action : np.int64 (0 ~ boardsize*2 - 1)
    '''
    def __init__(self,board_size = 19):
        self.action_space = gym.spaces.Discrete(board_size*board_size)
        self.observation_space = gym.spaces.Box(low=-1,high=1,shape=(2,board_size,board_size),dtype=np.float32)

        self.board_size = board_size
        self.board = []
        for _ in range(board_size):
            temp = ['.' for _ in range(self.board_size)]
            self.board.append(temp)
        self.turn = 'O'


        self._legal_moves = []
        for row in range(1,self.board_size+1):
            for col in range(1,self.board_size+1):
                self._legal_moves.append((row,col))
        self._legal_actions = [self.encode(move) for move in self._legal_moves]

    def reset(self):
        self.board = []
        for _ in range(self.board_size):
            temp = ['.' for _ in range(self.board_size)]
            self.board.append(temp)
        self.turn = 'O'
        observation = self.board_encoding(self.board,self.turn)


        self._legal_moves = []
        for row in range(1,self.board_size+1):
            for col in range(1,self.board_size+1):
                self._legal_moves.append((row,col))
        self._legal_actions = [self.encode(move) for move in self._legal_moves]
        return observation


    def step(self,action):
        observation = None
        reward = None
        DONE = False
        info = None
        if not self.is_possible_action(action):
            raise Exception('불가능한 action입니다.')

        # 가능한 action에서 이미 두어진 수는 제외합니다.
        move = self.decode(action)
        self._legal_moves.remove(move)
        self._legal_actions.remove(action)

        # 금지수를 입력하면 해당 턴에 해당하는 플레이어는 패배합니다.
        if not self.is_legal_action(action):
            reward = -1 if self.turn == 'O' else 1
            DONE = True

            x_move, y_move = self.decode(action)
            x = x_move - 1
            y = y_move - 1
            self.board[x][y] = self.turn
            self.turn = 'X' if self.turn == 'O' else 'O'

            observation = self.board_encoding(self.board,self.turn)
            return observation, reward, DONE, info

        # action을 진행합니다.
        x_move,y_move = self.decode(action)
        x = x_move - 1
        y = y_move - 1
        self.board[x][y] = self.turn

        # 경기 상태와 보상을 결정합니다.
        if self.check_5(x,y,self.board):
            reward = 1 if self.turn == 'O' else -1
            DONE = True
        else:
            reward = 0

        if self._legal_actions == [] and not DONE:
            DONE = True
            reward = 0

        self.turn = 'X' if self.turn == 'O' else 'O'

        observation = self.board_encoding(self.board, self.turn)
        return observation, reward, DONE, info

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
                row_string = row_string + ' ' + self.board[row][col] + ' '
            print(row_string)


    def legal_actions(self):
        return self._legal_actions

    def legal_moves(self):
        return self._legal_moves

    def encode(self,move):
        x,y = move
        return (x-1) * self.board_size + (y - 1)

    def decode(self,action):
        x = action // self.board_size + 1
        y = action % self.board_size + 1
        return x,y

    def check_in_board(self,action,check_board = None):
        x_move,y_move = self.decode(action)
        x = x_move - 1
        y = y_move - 1
        if (x in range(self.board_size)) and (y in range(self.board_size)):
            in_board = True
        else:
            in_board = False
        return x,y,in_board

    def check_is_empty(self,x,y,check_board = None):
        return True if check_board[x][y] == '.' else False

    def check_turn(self,check_board = None):
        return self.turn

    def check_5(self,check_x,check_y,check_board = None):
        check_color = self.turn
        max_strike = 0
        # 가로 체크
        letf_lim = max(0,check_y-5)
        for y_start in range(letf_lim,check_y + 1):
            strike = 0
            for ii in range(6):
                y = y_start + ii
                if y >= self.board_size:
                    break
                if check_board[check_x][y] == check_color:
                    strike += 1
                    max_strike = max(max_strike,strike)
                else:
                    break
        # 세로 체크
        upper_lim = max(0,check_x-5)
        for x_start in range(upper_lim,check_x+1):
            strike = 0
            for ii in range(6):
                x = x_start + ii
                if x >= self.board_size:
                    break
                if check_board[x][check_y] == check_color:
                    strike += 1
                    max_strike = max(max_strike,strike)
                else:
                    break
        # 좌하향 대각 체크
        x_d = check_x - max(0,check_x-5)
        y_d = check_y - max(0,check_y-5)
        distance = min(x_d,y_d)
        for d in range(0,distance+1):
            strike = 0
            x_start = check_x - d
            y_start = check_y - d
            for ii in range(6):
                x = x_start + ii
                y = y_start + ii

                if x >= self.board_size or y >= self.board_size:
                    break
                if check_board[x][y] == check_color:
                    strike += 1
                    max_strike = max(max_strike, strike)
                else:
                    break
        # 우상향 대각 체크
        x_d = min(self.board_size-1,check_x + 5) - check_x
        y_d = check_y - max(0,check_y-5)
        distance = min(x_d,y_d)
        for d in range(0,distance+1):
            strike = 0
            x_start = check_x + d
            y_start = check_y - d
            for ii in range(6):
                x = x_start - ii
                y = y_start + ii
                if x < 0 or y >= self.board_size:
                    break
                if check_board[x][y] == check_color:
                    strike += 1
                    max_strike = max(max_strike, strike)
                else:
                    break

        if max_strike == 5:
            return True
        return False



    def check_long(self,check_x,check_y,check_board = None):
        check_color = self.turn
        max_strike = 0
        # 가로 체크
        letf_lim = max(0, check_y - 5)
        for y_start in range(letf_lim, check_y + 1):
            strike = 0
            for ii in range(6):
                y = y_start + ii
                if y >= self.board_size:
                    break
                if check_board[check_x][y] == check_color:
                    strike += 1
                    max_strike = max(max_strike, strike)
                else:
                    break
        # 세로 체크
        upper_lim = max(0, check_x - 5)
        for x_start in range(upper_lim, check_x + 1):
            strike = 0
            for ii in range(6):
                x = x_start + ii
                if x >= self.board_size:
                    break
                if check_board[x][check_y] == check_color:
                    strike += 1
                    max_strike = max(max_strike, strike)
                else:
                    break
        # 좌하향 대각 체크
        x_d = check_x - max(0, check_x - 5)
        y_d = check_y - max(0, check_y - 5)
        distance = min(x_d, y_d)
        for d in range(0, distance + 1):
            strike = 0
            x_start = check_x - d
            y_start = check_y - d
            for ii in range(6):
                x = x_start + ii
                y = y_start + ii
                if x >= self.board_size or y >= self.board_size:
                    break
                if check_board[x][y] == check_color:
                    strike += 1
                    max_strike = max(max_strike, strike)
                else:
                    break
        # 우상향 대각 체크
        x_d = min(self.board_size-1, check_x + 5) - check_x
        y_d = check_y - max(0, check_y - 5)
        distance = min(x_d, y_d)
        for d in range(0, distance + 1):
            strike = 0
            x_start = check_x + d
            y_start = check_y - d
            for ii in range(6):
                x = x_start - ii
                y = y_start + ii
                if x < 0 or y >= self.board_size:
                    break
                if check_board[x][y] == check_color:
                    strike += 1
                    max_strike = max(max_strike, strike)
                else:
                    break
        if max_strike > 5:
            return True
        return False

    def check_44(self,check_x,check_y,check_board = None):
        count_4 = 0
        opponent_color = 'X' if self.turn == 'O' else 'O'
        # 가로 방향 44 체크
        y = check_y
        find_stones = 0
        while True:
            if y >= self.board_size or y < 0:
                break
            if check_board[check_x][y] == opponent_color or check_board[check_x][y] == '.':
                break
            if check_board[check_x][y] == self.turn:
                find_stones += 1
            y += 1

        y = check_y
        while True:
            if y >= self.board_size or y < 0:
                break
            if check_board[check_x][y] == opponent_color or check_board[check_x][y] == '.':
                break
            if check_board[check_x][y] == self.turn:
                find_stones += 1
            y -= 1

        if find_stones == 5:
            count_4 += 1

        # 세로 방향 44 체크
        x = check_x
        find_stones = 0
        while True:
            if x >= self.board_size or x < 0:
                break
            if (check_board[x][check_y] == opponent_color) or (check_board[x][check_y] == '.'):
                break
            if check_board[x][check_y] == self.turn:
                find_stones += 1
            x += 1

        x = check_x
        while True:
            if x >= self.board_size or x < 0:
                break
            if (check_board[x][check_y] == opponent_color) or (check_board[x][check_y] == '.'):
                break
            if check_board[x][check_y] == self.turn:
                find_stones += 1
            x -= 1

        if find_stones == 5:
            count_4 += 1

        # 좌하향 대각 44 체크
        x = check_x
        y = check_y
        find_stones = 0
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                break
            if (check_board[x][y] == opponent_color) or (check_board[x][y] == '.'):
                break
            if check_board[x][y] == self.turn:
                find_stones += 1
            x += 1
            y += 1

        x = check_x
        y = check_y
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                break
            if (check_board[x][y] == opponent_color) or (check_board[x][y] == '.'):
                break
            if check_board[x][y] == self.turn:
                find_stones += 1
            x -= 1
            y -= 1

        if find_stones == 5:
            count_4 += 1

        # 우상향 대각 44 체크
        x = check_x
        y = check_y
        find_stones = 0
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                break
            if (check_board[x][y] == opponent_color) or (check_board[x][y] == '.'):
                break
            if check_board[x][y] == self.turn:
                find_stones += 1
            x -= 1
            y += 1

        x = check_x
        y = check_y
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                break
            if (check_board[x][y] == opponent_color) or (check_board[x][y] == '.'):
                break
            if check_board[x][y] == self.turn:
                find_stones += 1
            x += 1
            y -= 1

        if find_stones == 5:
            count_4 += 1

        if count_4 >= 2:
            return True
        return False


    def check_33(self,check_x,check_y,check_board = None):
        count_3 = 0
        opponent_color = 'X' if self.turn == 'O' else 'O'
        # 가로 방향 33 체크
        y = check_y
        empty = 2
        find_stones = 0
        OPEN_L = True
        AROUND_STONE = False
        while True:
            if y >= self.board_size or y < 0:
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[check_x][y] == opponent_color:
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[check_x][y] == self.turn:
                find_stones += 1
                AROUND_STONE = True
            elif check_board[check_x][y] == '.':
                empty -= 1
                AROUND_STONE = False
            if empty <= 0:
                break
            y += 1

        y = check_y
        empty = 2 if find_stones == 1 else 1
        OPEN_R = True
        AROUND_STONE = False
        while True:
            if y >= self.board_size or y < 0:
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[check_x][y] == opponent_color:
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[check_x][y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[check_x][y] == '.':
                AROUND_STONE = False
                OPEN_R = True
                empty -= 1
            if empty <= 0:
                break
            y -= 1

        if find_stones == 4 and (OPEN_L and OPEN_R):
            count_3 += 1

        # 세로 방향 33 체크
        x = check_x
        empty = 2
        find_stones = 0
        OPEN_L = True
        AROUND_STONE = False
        while True:
            if x >= self.board_size or x < 0:
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[x][check_y] == opponent_color:
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[x][check_y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[x][check_y] == '.':
                AROUND_STONE = False
                empty -= 1
            if empty <= 0:
                break
            x += 1

        x = check_x
        empty = 2 if find_stones == 1 else 1
        OPEN_R = True
        AROUND_STONE = False
        while True:
            if x >= self.board_size or x < 0:
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[x][check_y] == opponent_color:
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[x][check_y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[x][check_y] == '.':
                AROUND_STONE = False
                empty -= 1
            if empty <= 0:
                break
            x -= 1

        if find_stones == 4 and (OPEN_L and OPEN_R):
            count_3 += 1

        # 좌하향 대각 33 체크
        x = check_x
        y = check_y
        empty = 2
        find_stones = 0
        OPEN_L = True
        AROUND_STONE = False
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[x][y] == opponent_color:
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[x][y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[x][y] == '.':
                AROUND_STONE = True
                empty -= 1
            if empty <= 0:
                break
            x += 1
            y += 1

        x = check_x
        y = check_y
        empty = 2 if find_stones == 1 else 1
        OPEN_R = True
        AROUND_STONE = False
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[x][y] == opponent_color:
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[x][y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[x][y] == '.':
                AROUND_STONE = True
                empty -= 1
            if empty <= 0:
                break
            x -= 1
            y -= 1

        if find_stones == 4 and (OPEN_L and OPEN_R):
            count_3 += 1

        # 우상향 대각 33 체크
        x = check_x
        y = check_y
        empty = 2
        find_stones = 0
        OPEN_L = True
        AROUND_STONE = False
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[x][y] == opponent_color:
                OPEN_L = False if AROUND_STONE else True
                break
            if check_board[x][y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[x][y] == '.':
                AROUND_STONE = True
                empty -= 1
            if empty <= 0:
                break
            x -= 1
            y += 1

        x = check_x
        y = check_y
        empty = 2 if find_stones == 1 else 1
        OPEN_R = True
        AROUND_STONE = False
        while True:
            if (x >= self.board_size or x < 0) or (y >= self.board_size or y < 0):
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[x][y] == opponent_color:
                OPEN_R = False if AROUND_STONE else True
                break
            if check_board[x][y] == self.turn:
                AROUND_STONE = True
                find_stones += 1
            elif check_board[x][y] == '.':
                AROUND_STONE = True
                empty -= 1
            if empty <= 0:
                break
            x += 1
            y -= 1

        if find_stones == 4 and (OPEN_L and OPEN_R):
            count_3 += 1

        if count_3 >= 2:
            return True
        return False


    def is_legal_action(self,action):
        # 판단하기 위한 참고용 보드입니다.
        check_board = copy.deepcopy(self.board)
        LEGAL_ACTION = True
        x_move,y_move = self.decode(action)
        x = x_move - 1
        y = y_move - 1

        # 백 턴에는 금지수가 없습니다.
        if self.turn == 'X':
            return True

        # 참고용 보드에 수를 진행해 보고 금지수인지 판단해봅니다.
        check_board[x][y] = self.turn

        # 금지수더라도, 그 수로 인해서 5가 완성되면 허용됩니다.
        if self.check_5(x, y,check_board):
            return True

        # 장목은 금지됩니다.
        if self.check_long(x,y,check_board):
            return False

        # 33이 금지됩니다.
        if self.check_33(x,y,check_board):
            return False

        # 44가 금지됩니다.
        if self.check_44(x,y,check_board):
            return False
        return LEGAL_ACTION

    def is_possible_action(self,action):
        x, y, IS_IN_BOARD = self.check_in_board(action)
        if not IS_IN_BOARD:
            print('보드의 범위를 벗어나는 action이 입력되었습니다. \n (x,y) : ({:d},{:d})'.format(x + 1, y + 1))
            return False

        if not self.check_is_empty(x, y,self.board):
            self.render()
            print('해당 칸에 이미 돌이 놓여져 있습니다.\n 좌표 : ({:d},{:d})'.format(x + 1, y + 1))
            return False
        return True

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

    def board_decoding(self,observation):
        turn = 'O' if observation[0][0][1] == 1 else 'X'
        board_size = observation.shape[0]

        board = []
        for _ in range(board_size):
            temp = ['.' for _ in range(board_size)]
            board.append(temp)

        for raw in range(board_size):
            for col in range(board_size):
                if observation[raw][col][0] == 1:
                    stone = 'O'
                elif observation[raw][col][0] == -1:
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


class Omok_Ver2(gym.Env):
    '''
    white = -1, black = 1, empty = 0
    observation : np.ndarray (2,board_size,board_size)
    action : np.int64 (0 ~ boardsize*2 - 1)
    '''
    def __init__(self,board_size = 19):
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
        action_type = c_func(self.board_array,action,self.turn,self.board_size)

        if action_type == 0:
            ''' 금지수가 아니고 승부가 나지 않은 상황입니다. '''
            done = False
            if not self.legal_actions:
                done = True # 무승부가 난 경우입니다.
            reward = 0.

        elif action_type == 1:
            ''' 금지수가 입력된 경우입니다. '''
            done = True
            reward = 1. if self.turn == -1 else -1.

        else:
            ''' 금지수가 아닌 경우로 승리한 경우입니다.'''
            done = True
            reward = 1. if self.turn == 1 else -1.

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

    def board_decoding(self,observation):
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

class OmokVer3(gym.Env):
    '''
    white = -1, black = 1, empty = 0
    observation : np.ndarray (2,board_size,board_size)
    action : np.int64 (0 ~ boardsize*2 - 1)
    '''
    def __init__(self,board_size = 19):
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
        print(f'Action type : {action_type}')
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

    def board_decoding(self,observation):
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


if __name__ == '__main__':
    import random
    env = OmokVer3()
    obs = env.reset()
    for ii in range(300):
        move = input("Enter the coordinate : ").split(' ')
        move = list(map(int,move))
        action = env.encode_action(move)
        if action not in env.legal_actions:
            continue
        obs,reward,done,_ = env.step(action)
        print(ii,env.decode_action(action),reward)
        env.board_decoding(obs)
        if done:
            break
