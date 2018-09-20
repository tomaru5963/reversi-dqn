from reversi_board import ReversiBoard


class ReversiGame(object):

    def __init__(self, display=False):
        self.display = display
        self.board = ReversiBoard(self.display)

    def reset(self):
        self.board.reset()

    def play(self, player1, player2):
        players = {ReversiBoard.PLAYER_X: player1,
                   ReversiBoard.PLAYER_O: player2}

        if self.display:
            print(f'X: {players[ReversiBoard.PLAYER_X].name}, '
                  f'O: {players[ReversiBoard.PLAYER_O].name}')

        for player in (ReversiBoard.PLAYER_X, ReversiBoard.PLAYER_O):
            players[player].on_game_started(player)

        player = ReversiBoard.PLAYER_X
        while self.board.state == ReversiBoard.ACTIVE:
            pos = players[player].on_next_move_required(player)
            self.board.make_next_move(pos, player)

            if player == ReversiBoard.PLAYER_X:
                if len(self.board.available_places[ReversiBoard.PLAYER_O]) > 0:
                    player = ReversiBoard.PLAYER_O
            else:
                if len(self.board.available_places[ReversiBoard.PLAYER_X]) > 0:
                    player = ReversiBoard.PLAYER_X

        for player in (ReversiBoard.PLAYER_X, ReversiBoard.PLAYER_O):
            pos = players[player].on_next_move_required(player)
            assert pos is None
            players[player].on_game_finished(player)

        if self.display:
            print(f'X: {players[ReversiBoard.PLAYER_X].name}, '
                  f'O: {players[ReversiBoard.PLAYER_O].name}')

        return self.board.state
