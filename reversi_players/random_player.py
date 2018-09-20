import random

from .player_base import PlayerBase


class RandomPlayer(PlayerBase):

    def __init__(self, board, name='RandomPlayer', train=False):
        super(RandomPlayer, self).__init__(board, name, train)

    def on_next_move_required(self, who_am_i):
        if self.board.state != self.board.ACTIVE:
            return None

        places = self.board.available_places[who_am_i]
        pos = random.choice(list(places.keys()))
        return pos
