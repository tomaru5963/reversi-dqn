import argparse

import pygame

from reversi_board import ReversiBoard
from reversi_players.dqn_player import DQNPlayer
from reversi_players.greedy_player import GreedyPlayer
from reversi_players.player_base import PlayerBase
from reversi_players.random_player import RandomPlayer


TILE_SIZE = (96, 96)
BOARD_SIZE = (TILE_SIZE[0] * ReversiBoard.NUM_COLS,
              TILE_SIZE[1] * ReversiBoard.NUM_ROWS)
SCORE_AREA_SIZE = (BOARD_SIZE[0], TILE_SIZE[1] // 2)
MSG_AREA_SIZE = (BOARD_SIZE[0], TILE_SIZE[1] // 2)
FONT_SIZE = TILE_SIZE[1] // 3
WINDOW_SIZE = (BOARD_SIZE[0], BOARD_SIZE[1] + SCORE_AREA_SIZE[1] + MSG_AREA_SIZE[1])


class HumanPlayer(PlayerBase):

    def __init__(self, name='Human', train=False):
        super(HumanPlayer, self).__init__(name, train)


PLAYERS = {'dqn': DQNPlayer,
           'random': RandomPlayer,
           'greedy': GreedyPlayer,
           'human': HumanPlayer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', nargs='?', default='greedy')
    parser.add_argument('player2', nargs='?', default='human')
    parser.add_argument('--load1', metavar='FILE')
    parser.add_argument('--load2', metavar='FILE')
    args = parser.parse_args()

    board = ReversiBoard(display=False)
    player1 = PLAYERS[args.player1](board, train=False)
    player2 = PLAYERS[args.player2](board, train=False)

    if args.load1:
        player1.load_params(args.load1)
    if args.load2:
        player2.load_params(args.load2)

    pygame.init()
    GUIGame(board, player1, player2).play()
    pygame.quit()


class GUIGame(object):

    def __init__(self, board, player1, player2):
        self.players = {ReversiBoard.PLAYER_X: player1,
                        ReversiBoard.PLAYER_O: player2}
        self.who_am_i = ReversiBoard.PLAYER_X
        self.board = board

        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.font = pygame.font.Font(None, FONT_SIZE)
        images = self.build_images(self.screen.get_size())
        self.background = images['background']
        self.mouse_images = {ReversiBoard.PLAYER_X: images['cross'],
                             ReversiBoard.PLAYER_O: images['circle']}

        self.tiles = pygame.sprite.RenderUpdates()
        for row in range(ReversiBoard.NUM_ROWS):
            for col in range(ReversiBoard.NUM_COLS):
                self.tiles.add(self.Tile(col * TILE_SIZE[0],
                                         row * TILE_SIZE[1],
                                         images,
                                         (row, col), self.board))
        self.score_area_rect = pygame.Rect(
            (0, ReversiBoard.NUM_ROWS * TILE_SIZE[1]),
            SCORE_AREA_SIZE
        )
        self.msg_area_rect = pygame.Rect(
            self.score_area_rect.bottomleft,
            MSG_AREA_SIZE
        )

    def play(self):
        self.board.reset()
        for who_am_i in (ReversiBoard.PLAYER_X, ReversiBoard.PLAYER_O):
            self.players[who_am_i].on_game_started(who_am_i)

        clock = pygame.time.Clock()
        mouse_pos = (0, 0)
        is_running = True
        while is_running:
            clock.tick(30)

            place = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                else:
                    place = self.handle_event(event)
                    if event.type == pygame.MOUSEMOTION:
                        mouse_pos = event.pos

            self.make_next_move(place)
            self.update(mouse_pos)

        if self.board.state != ReversiBoard.ACTIVE:
            for who_am_i in (ReversiBoard.PLAYER_X, ReversiBoard.PLAYER_O):
                if not self.is_human(who_am_i):
                    place = self.players[who_am_i].on_next_move_required(who_am_i)
                    assert place is None
                self.players[who_am_i].on_game_finished(who_am_i)

    def is_human(self, who_am_i):
        return isinstance(self.players[who_am_i], HumanPlayer)

    def make_next_move(self, place):
        if self.board.state != ReversiBoard.ACTIVE:
            return

        if self.is_human(self.who_am_i):
            if place is None:
                return
        else:
            place = self.players[self.who_am_i].on_next_move_required(self.who_am_i)

        self.board.make_next_move(place, self.who_am_i)
        if self.who_am_i == ReversiBoard.PLAYER_X:
            if len(self.board.available_places[ReversiBoard.PLAYER_O]) > 0:
                self.who_am_i = ReversiBoard.PLAYER_O
        else:
            if len(self.board.available_places[ReversiBoard.PLAYER_X]) > 0:
                self.who_am_i = ReversiBoard.PLAYER_X

    def update(self, mouse_pos):
        self.tiles.update(self.who_am_i)
        self.screen.blit(self.background, (0, 0))
        self.tiles.draw(self.screen)

        # score
        text = self.font.render(f'Black {self.board.scores[ReversiBoard.WON_X]}, '
                                f'White {self.board.scores[ReversiBoard.WON_O]}',
                                True, (250, 250, 250))
        score_rect = text.get_rect()
        score_rect.center = self.score_area_rect.center
        self.screen.blit(text, score_rect)

        # status
        player = 'Black' if self.who_am_i == ReversiBoard.PLAYER_X else 'White'
        message = f"{player}: {self.players[self.who_am_i].name}'s turn"
        if self.board.state != ReversiBoard.ACTIVE:
            if self.board.state == ReversiBoard.WON_X:
                message = f'Black: {self.players[ReversiBoard.WON_X].name} won'
            elif self.board.state == ReversiBoard.WON_O:
                message = f'White: {self.players[ReversiBoard.WON_O].name} won'
            else:
                message = 'The game was drawn'
        text = self.font.render(message, True, (250, 250, 250))
        message_rect = text.get_rect()
        message_rect.center = self.msg_area_rect.center
        self.screen.blit(text, message_rect)

        if (self.is_human(self.who_am_i) and
                self.board.state == ReversiBoard.ACTIVE):
            image = self.mouse_images[self.who_am_i]
            rect = image.get_rect()
            rect.center = mouse_pos
            self.screen.blit(image, rect)

        pygame.display.update()

    def handle_event(self, event):
        place = None
        for tile in self.tiles:
            place = tile.handle_event(event, self.who_am_i) if place is None else place
        return place

    @staticmethod
    def build_images(screen_size):
        BLACK = (10, 10, 10)
        WHITE = (250, 250, 250)
        GREEN = (0, 128, 0)

        images = {}
        image = pygame.Surface(TILE_SIZE, flags=pygame.SRCALPHA)
        rect = image.get_rect()

        # cross (black)
        images['cross'] = image.copy()
        temp_rect = rect.copy()
        temp_rect.width = int(temp_rect.width * .7)
        temp_rect.height = int(temp_rect.height * .7)
        temp_rect.center = rect.center
        pygame.draw.circle(images['cross'], BLACK, temp_rect.center, temp_rect.width // 2)

        # circle (white)
        images['circle'] = image.copy()
        pygame.draw.circle(images['circle'], WHITE, temp_rect.center, temp_rect.width // 2)

        # empty tile
        images['empty_tile'] = image.copy()
        pygame.draw.rect(images['empty_tile'], WHITE, rect, 2)

        # placeable tile
        images['placeable_tile'] = images['empty_tile'].copy()
        images['placeable_tile'].fill((255, 255, 255, 128))

        # cross tile
        images['cross_tile'] = images['empty_tile'].copy()
        images['cross_tile'].blit(images['cross'], (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # circle tile
        images['circle_tile'] = images['empty_tile'].copy()
        images['circle_tile'].blit(images['circle'], (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # background
        images['background'] = pygame.Surface(screen_size)
        images['background'].fill(GREEN)

        return images

    class Tile(pygame.sprite.Sprite):

        def __init__(self, x, y, images, place, board):
            super(GUIGame.Tile, self).__init__()
            self.images = images
            self.image = self.images['empty_tile']
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
            self.is_mouse_on = False

            self.place = place
            self.board = board

        def handle_event(self, event, player):
            self.is_mouse_on = False
            if (event.type == pygame.MOUSEMOTION and
                    self.rect.collidepoint(event.pos)):
                self.is_mouse_on = True

            place = None
            if (event.type == pygame.MOUSEBUTTONUP and
                    self.place in self.board.available_places[player] and
                    self.rect.collidepoint(event.pos)):
                place = self.place
            return place

        def update(self, player):
            if self.board.board[self.place] == ReversiBoard.PLAYER_X:
                self.image = self.images['cross_tile']
            elif self.board.board[self.place] == ReversiBoard.PLAYER_O:
                self.image = self.images['circle_tile']
            # if self.place in self.board.available_places[player]:
            else:
                if (self.is_mouse_on and
                        self.place in self.board.available_places[player]):
                    self.image = self.images['placeable_tile']
                else:
                    self.image = self.images['empty_tile']


if __name__ == '__main__':
    main()
