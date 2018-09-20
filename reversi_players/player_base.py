class PlayerBase(object):

    def __init__(self, board, name, train):
        self.board = board
        self.name = name
        self.train = train

    def on_game_started(self, who_am_i):
        pass

    def on_game_finished(self, who_am_i):
        pass

    def on_next_move_required(self, who_am_i):
        raise NotImplementedError

    def load_params(self, path):
        pass

    def save_params(self, path):
        pass
