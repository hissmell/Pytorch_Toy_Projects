import numpy as np
cimport numpy as np

def check_action_type(float[:,:] board_array,int action,int turn,int board_size):
    cdef int check_x, check_y
    cdef int check_color
    cdef int max_strike, strike
    cdef int find_stones, opponent_color
    cdef int count_3, count_4
    cdef int HARD_BOUND_L, HARD_BOUND_R
    cdef int AROUND_STONE, HARD_BOUND, WEAK_BOUND_L, WEAK_BOUND_R

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
    HARD_BOUND_L = 0
    while 1:
        if (y >= board_size or y < 0) or (board_array[check_x][y] == opponent_color):
            HARD_BOUND_L = 1
            break
        if board_array[check_x][y] == 0:
            break
        if board_array[check_x][y] == turn:
            strike += 1
        y += 1

    y = check_y - 1
    HARD_BOUND_R = 0
    while 1:
        if (y >= board_size or y < 0) or (board_array[check_x][y] == opponent_color):
            HARD_BOUND_R = 1
            break
        if board_array[check_x][y] == 0:
            break
        if board_array[check_x][y] == turn:
            strike += 1
        y -= 1

    if strike == 4:
        if not (HARD_BOUND_L and HARD_BOUND_R):
            count_4 += 1

    if max_strike < strike:
        max_strike = strike

    # 세로 체크
    x = check_x + 1
    strike = 1
    HARD_BOUND_L = 0
    while 1:
        if (x >= board_size or x < 0) or (board_array[x][check_y] == opponent_color):
            HARD_BOUND_L = 1
            break
        if  board_array[x][check_y] == 0:
            break
        if board_array[x][check_y] == turn:
            strike += 1
        x += 1

    HARD_BOUND_R = 0
    x = check_x - 1
    while 1:
        if (x >= board_size or x < 0) or (board_array[x][check_y] == opponent_color):
            HARD_BOUND_R = 1
            break
        if board_array[x][check_y] == 0:
            break
        if board_array[x][check_y] == turn:
            strike += 1
        x -= 1

    if strike == 4:
        if not (HARD_BOUND_L and HARD_BOUND_R):
            count_4 += 1

    if max_strike < strike:
        max_strike = strike

    # 좌하향 대각 체크
    x = check_x + 1
    y = check_y - 1
    strike = 1
    HARD_BOUND_L = 0
    while 1:
        if ((x >= board_size or x < 0) or (y >= board_size or y < 0)) or (board_array[x][y] == opponent_color):
            HARD_BOUND_L = 1
            break
        if  board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x += 1
        y -= 1

    x = check_x - 1
    y = check_y + 1
    HARD_BOUND_R = 0
    while 1:
        if ((x >= board_size or x < 0) or (y >= board_size or y < 0)) or (board_array[x][y] == opponent_color):
            HARD_BOUND_R = 1
            break
        if board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x -= 1
        y += 1

    if strike == 4:
        if not (HARD_BOUND_L and HARD_BOUND_R):
            count_4 += 1

    if max_strike < strike:
        max_strike = strike

    # 우하향 대각 체크
    x = check_x - 1
    y = check_y - 1
    strike = 1
    HARD_BOUND_L = 0
    while 1:
        if ((x >= board_size or x < 0) or (y >= board_size or y < 0)) or (board_array[x][y] == opponent_color):
            HARD_BOUND_L = 1
            break
        if board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x -= 1
        y -= 1

    x = check_x + 1
    y = check_y + 1
    HARD_BOUND_R = 0
    while 1:
        if ((x >= board_size or x < 0) or (y >= board_size or y < 0)) or (board_array[x][y] == opponent_color):
            HARD_BOUND_R = 1
            break
        if board_array[x][y] == 0:
            break
        if board_array[x][y] == turn:
            strike += 1
        x += 1
        y += 1

    if strike == 4:
        if not (HARD_BOUND_L and HARD_BOUND_R):
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
