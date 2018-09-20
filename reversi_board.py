import itertools

import numpy as np


class ReversiBoard(object):

    NUM_ROWS = 4
    NUM_COLS = 4

    EMPTY = 0
    PLAYER_X = 1
    PLAYER_O = 2

    ACTIVE = 0
    WON_X = 1
    WON_O = 2
    DRAW = 3

    def __init__(self, display=True):
        self.display = display
        self.board = np.zeros((self.NUM_ROWS, self.NUM_COLS), dtype=np.int32)
        self.state = self.ACTIVE
        self.available_places = {self.PLAYER_X: {}, self.PLAYER_O: {}}
        self.scores = {self.PLAYER_X: 0, self.PLAYER_O: 0}
        self.reset()

    def reset(self):
        self.board[:, :] = self.EMPTY
        self.state = self.ACTIVE
        self.available_places[self.PLAYER_X].clear()
        self.available_places[self.PLAYER_O].clear()
        self.scores[self.PLAYER_X] = 0
        self.scores[self.PLAYER_O] = 0

        self.board[self.NUM_ROWS // 2 - 1, self.NUM_COLS // 2] = self.PLAYER_X
        self.board[self.NUM_ROWS // 2, self.NUM_COLS // 2 - 1] = self.PLAYER_X
        self.board[self.NUM_ROWS // 2 - 1, self.NUM_COLS // 2 - 1] = self.PLAYER_O
        self.board[self.NUM_ROWS // 2, self.NUM_COLS // 2] = self.PLAYER_O
        self.update_state()
        if self.display:
            self.show()

    def get_available_places(self, player):
        rows, cols = np.where(self.board == self.EMPTY)
        places = {}
        for pos in zip(rows, cols):
            discs = self.find_flippable_discs(pos, player)
            if len(discs) > 0:
                places[pos] = discs
        return places

    def find_flippable_discs(self, pos, player):
        discs = []
        for direction in itertools.product((-1, 0, 1), repeat=2):
            if direction[0] == 0 and direction[1] == 0:
                continue
            discs.extend(self.find_flippable_line(pos, direction, player))
        return discs

    def find_flippable_line(self, pos, direction, player):
        pos = np.array(pos)
        direction = np.array(direction)
        opposite_player = self.PLAYER_O
        if player == self.PLAYER_O:
            opposite_player = self.PLAYER_X

        discs = []
        pos += direction
        while True:
            if (pos[0] < 0 or pos[1] < 0 or
                    pos[0] >= self.NUM_ROWS or pos[1] >= self.NUM_COLS or
                    self.board[tuple(pos)] == self.EMPTY):
                discs = []
                break
            elif self.board[tuple(pos)] == player:
                break
            assert self.board[tuple(pos)] == opposite_player
            discs.append(tuple(pos))
            pos += direction
        return discs

    def make_next_move(self, pos, player):
        assert pos in self.available_places[player]

        self.board[pos] = player
        # flip discs
        discs = self.available_places[player][pos]
        self.board[tuple(zip(*discs))] = player

        self.update_state()
        if self.display:
            self.show()

    def update_state(self):
        for player in (self.PLAYER_X, self.PLAYER_O):
            self.available_places[player] = self.get_available_places(player)
            self.scores[player] = np.sum(self.board == player)

        if (len(self.available_places[self.PLAYER_X]) == 0 and
                len(self.available_places[self.PLAYER_O]) == 0):
            if self.scores[self.PLAYER_X] > self.scores[self.PLAYER_O]:
                self.state = self.WON_X
            elif self.scores[self.PLAYER_X] < self.scores[self.PLAYER_O]:
                self.state = self.WON_O
            else:
                self.state = self.DRAW
        else:
            self.state = self.ACTIVE

    def show(self):
        head_tail = '----' * self.NUM_COLS + '-'
        state_message = {self.ACTIVE: f'{self.scores}',
                         self.WON_X: f'Won by X, {self.scores}',
                         self.WON_O: f'Won by O, {self.scores}',
                         self.DRAW: f'Drawn, {self.scores}'}

        for row in self.board:
            print(head_tail)
            for cell in row:
                if cell == self.EMPTY:
                    print('|   ', end='')
                elif cell == self.PLAYER_X:
                    print('| X ', end='')
                else:
                    print('| O ', end='')
            print('|')
        print(head_tail, state_message[self.state])
