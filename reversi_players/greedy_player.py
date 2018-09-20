from .player_base import PlayerBase


class GreedyPlayer(PlayerBase):

    def __init__(self, board, name='GreedyPlayer', train=False):
        super(GreedyPlayer, self).__init__(board, name, train)

    def on_next_move_required(self, who_am_i):
        if self.board.state != self.board.ACTIVE:
            return None

        max_discs = 0
        best_pos = None
        for pos, discs in self.board.available_places[who_am_i].items():
            num_discs = len(discs)
            if num_discs > max_discs:
                best_pos = pos
                max_discs = num_discs
        return best_pos
