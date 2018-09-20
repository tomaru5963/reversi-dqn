import argparse

import matplotlib.pyplot as plt
import numpy as np

from reversi_board import ReversiBoard
from reversi_game import ReversiGame
from reversi_players.dqn_player import DQNPlayer
from reversi_players.greedy_player import GreedyPlayer
from reversi_players.random_player import RandomPlayer


PLAYERS = {'dqn': DQNPlayer,
           'random': RandomPlayer,
           'greedy': GreedyPlayer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', nargs='?', default='random')
    parser.add_argument('player2', nargs='?', default='random')
    parser.add_argument('num_matches', nargs='?', type=int, default=10000)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--load1', metavar='FILE')
    parser.add_argument('--load2', metavar='FILE')
    parser.add_argument('--save1', metavar='FILE')
    parser.add_argument('--save2', metavar='FILE')
    parser.add_argument('--train1', action='store_true')
    parser.add_argument('--train2', action='store_true')
    args = parser.parse_args()

    game = ReversiGame(display=args.display)

    player1 = PLAYERS[args.player1](game.board, train=args.train1)
    player2 = PLAYERS[args.player2](game.board, train=args.train2)
    summary = {player1: {'wins': 0, 'rates': []},
               player2: {'wins': 0, 'rates': []},
               'draw': {'wins': 0, 'rates': []}}

    if args.load1:
        player1.load_params(args.load1)
    if args.load2:
        player2.load_params(args.load2)

    for match in range(args.num_matches):
        if np.random.randint(2) == 0:
            p1 = player1
            p2 = player2
        else:
            p1 = player2
            p2 = player1

        game.reset()
        result = game.play(p1, p2)

        if result == ReversiBoard.WON_X:
            summary[p1]['wins'] += 1
        elif result == ReversiBoard.WON_O:
            summary[p2]['wins'] += 1
        else:
            summary['draw']['wins'] += 1
        for elem in summary.values():
            elem['rates'].append(elem['wins'] / (match + 1))

        print(f"{match + 1}/{args.num_matches} "
              f"{player1.name:10.10}: {summary[player1]['rates'][-1]:1.2f}, "
              f"{player2.name:10.10}: {summary[player2]['rates'][-1]:1.2f}, "
              f"Draw: {summary['draw']['rates'][-1]:1.2f}")

    if args.save1:
        player1.save_params(args.save1)
    if args.save2:
        player2.save_params(args.save2)

    print(f"""
    {player1.name:15.15}: {summary[player1]['rates'][-1]}
    {player2.name:15.15}: {summary[player2]['rates'][-1]}
    Draw           : {summary['draw']['rates'][-1]}
    """)

    plt.plot(summary[player1]['rates'], label=player1.name)
    plt.plot(summary[player2]['rates'], label=player2.name)
    plt.plot(summary['draw']['rates'], label='Draw')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
