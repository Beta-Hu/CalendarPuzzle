from typing import List
import numpy as np
from copy import deepcopy
import time
import random
import threading
import argparse

global FINISH
FINISH = False

class Calendar:
    MONTH = [   [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    DAY =  [    [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
                [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6],
                [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6],
                [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],
                [6, 0], [6, 1], [6, 2]]
    WEEK = [    [6, 4], [6, 5], [6, 6], [7, 4], [7, 5], [7, 6], [6, 3]]


class Block:
    def __init__(self, num_flip, num_rotate) -> None:
        self.data = np.zeros([8, 8])
        self.useable = True
        self.num_flip = num_flip
        self.num_rotate = num_rotate
        self.id = -1

    def flip(self):
        check = np.sum(self.data)
        r = np.where(np.sum(self.data, axis=1) > 0)
        c = np.where(np.sum(self.data, axis=0) > 0)
        R, C = np.meshgrid(r, c)
        tmp = self.data
        self.data = 0 * self.data
        self.data[R, C] = tmp[R, C[::-1]]
        if np.sum(self.data) != check:
            self.data = tmp
        return self
    
    def rotate(self, num_rotate):
        check = np.sum(self.data)
        R_MIN, R_MAX = np.min(np.where(np.sum(self.data, axis=1))), np.max(np.where(np.sum(self.data, axis=1)))
        C_MIN, C_MAX = np.min(np.where(np.sum(self.data, axis=0))), np.max(np.where(np.sum(self.data, axis=0)))
        S = np.max([R_MAX - R_MIN + 1, C_MAX - C_MIN + 1])
        R, C = np.meshgrid(np.arange(R_MIN, R_MIN + S), np.arange(C_MIN, C_MIN + S))
        if np.max(R) < 8 and np.max(C) < 8:
            tmp = self.data
            self.data = 0 * self.data
            self.data[C, R] = np.rot90(tmp[C, R], num_rotate)
        if np.sum(self.data) != check:
            return None
        while np.sum(self.data, axis=0)[0] == 0:
            self.data = self.data[:, [1, 2, 3, 4, 5, 6, 7, 0]]
        while np.sum(self.data, axis=1)[0] == 0:
            self.data = self.data[[1, 2, 3, 4, 5, 6, 7, 0], :]
        return self

    def move_lr(self):
        if np.sum(self.data, axis=0)[-1] == 0:
            self.data = self.data[:, [7, 0, 1, 2, 3, 4, 5, 6]]
            return 1
        return 0

    def move_ud(self):
        if np.sum(self.data, axis=1)[-1] == 0:
            self.data = self.data[[7, 0, 1, 2, 3, 4, 5, 6], :]
            return 1
        return 0
    
    def reset_lr(self):
        while np.sum(self.data, axis=0)[0] == 0:
            self.data = self.data[:, [1, 2, 3, 4, 5, 6, 7, 0]]

    def reset_ud(self):
        while np.sum(self.data, axis=1)[0] == 0:
            self.data = self.data[[1, 2, 3, 4, 5, 6, 7, 0], :]

def forward(board):
    b0 = Block(0, 2)
    b0.data[[0, 0, 0, 0], [0, 1, 2, 3]] = 10
    b1 = Block(1, 2)
    b1.data[[0, 1, 1, 2], [0, 0, 1, 1]] = 1
    b2 = Block(1, 4)
    b2.data[[0, 0, 1, 1, 1], [0, 1, 1, 2, 3]] = 2
    b3 = Block(1, 4)
    b3.data[[0, 1, 1, 1, 1], [0, 0, 1, 2, 3]] = 3
    b4 = Block(0, 4)
    b4.data[[0, 0, 1, 2, 2], [0, 1, 1, 0, 1]] = 4
    b5 = Block(1, 4)
    b5.data[[0, 1, 1, 1], [2, 0, 1, 2]] = 5
    b6 = Block(1, 4)
    b6.data[[0, 0, 1, 1, 2], [0, 1, 0, 1, 0]] = 6
    b7 = Block(0, 4)
    b7.data[[0, 0, 0, 1, 2], [0, 1, 2, 0, 0]] = 7
    b8 = Block(1, 2)
    b8.data[[0, 1, 2, 0, 2], [1, 1, 1, 2, 0]] = 8
    b9 = Block(0, 4)
    b9.data[[0, 1, 2, 1, 1], [0, 0, 0, 1, 2]] = 9

    blocks = []
    for b in [b9, b1, b2, b3, b4, b5, b6, b7, b8, b0]:
        for i in range(0, b.num_rotate):
            tmp_b = deepcopy(b).rotate(i)
            tmp_b.id = np.max(tmp_b.data)
            if tmp_b is not None:
                blocks.append(tmp_b)
                for _ in range(tmp_b.num_flip):
                    blocks.append(deepcopy(tmp_b).flip())

    random.shuffle(blocks)
    print("Sub thread start.")
    # [print(b.data, '\n') for b in blocks]

    LEGAL = (board < 0).astype('float')
    st = time.time()
    put_one(board, blocks, LEGAL)
    print("\nTime: %.2f sec" % (time.time() - st))
    

def put_one(board, blocks: List[Block], LEGAL):
    global FINISH

    if not np.any(board == 0):
        print(board)
        FINISH = True
        return 1
    
    if FINISH:
        return 1

    x, y = np.where(board == 0)
    if np.any(connectivity(board, x[0], y[0]) == 0):
        return 0

    blocks_for_next = [deepcopy(b) for b in blocks]
    for block in blocks:
        if not block.useable:
            continue

        block.reset_lr()
        block.reset_ud()

        for i in range(8):
            # out of board ?
            # adjacency ?
            for j in range(8):
                # out of board ?
                # adjacency ?
                # stack ?
                if (j > 0):
                    block.move_lr()

                if np.any((block.data > 0) * (board > 0)):
                    continue
                # illegal location ?
                if np.any((block.data > 0) * (LEGAL > 0)):
                    continue
                
                for b in blocks_for_next:
                    if b.id == block.id:
                        b.useable = False

                isfinish = put_one(board + block.data, blocks_for_next, LEGAL)

                for b in blocks_for_next:
                    if b.id == block.id:
                        b.useable = True

                if isfinish:
                    return 1

            block.reset_lr()
            block.move_ud()
        block.reset_lr()
        block.reset_ud()


def connectivity(board, x, y):
    board = (board != 0).astype('float')
    board[x, y] = 1

    if x > 0:
        if board[x - 1, y] == 0:
            board = connectivity(board, x - 1, y)
    if y > 0:
        if board[x, y - 1] == 0:
            board = connectivity(board, x, y - 1)
    if x < 7:
        if board[x + 1, y] == 0:
            board = connectivity(board, x + 1, y)
    if y < 7:
        if board[x, y + 1] == 0:
            board = connectivity(board, x, y + 1)
    return board


if __name__ == '__main__':
    # init board

    parser = argparse.ArgumentParser()
    parser.add_argument('--month', '-m', type=int)
    parser.add_argument('--day', '-d', type=int)
    parser.add_argument('--week', '-w', type=int)
    parser.add_argument('--num_thread', '-t', type=int, default=16)
    args = parser.parse_args()

    M = args.month
    D = args.day
    W = args.week

    board = np.zeros([8, 8])
    board[:, -1] = - np.inf
    board[:2, -2] = - np.inf
    board[-1, :4] = - np.inf

    board[Calendar().MONTH[M-1][0], Calendar().MONTH[M-1][1]] = - np.inf
    board[Calendar().DAY[D-1][0], Calendar().DAY[D-1][1]] = - np.inf
    board[Calendar().WEEK[W-1][0], Calendar().WEEK[W-1][1]] = - np.inf

    thread_pool = [threading.Thread(target=forward, args=(deepcopy(board), )) for i in range(args.num_thread)]
    _ = [t.start() for t in thread_pool]
    _ = [t.join() for t in thread_pool]


