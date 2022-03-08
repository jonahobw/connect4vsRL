import logging
import argparse

import config
from model import Residual_CNN
from agent import Agent, PyTorchAgent
from funcs import playMatches
from game import Game

logger = logging.getLogger("foo")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

HIDDEN_CNN_LAYERS = [
    {"filters": 75, "kernel_size": (4, 4)},
    {"filters": 75, "kernel_size": (4, 4)},
    {"filters": 75, "kernel_size": (4, 4)},
    {"filters": 75, "kernel_size": (4, 4)},
    {"filters": 75, "kernel_size": (4, 4)},
    {"filters": 75, "kernel_size": (4, 4)},
]


def tourney(run_version=0, player1version=8, turns_until_tau=10, episodes=10, pytorch_first=False):
    env = Game()
    player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                              config.HIDDEN_CNN_LAYERS)

    if player1version > 0:
        player1_network = player1_NN.read(env.name, run_version, player1version)
        player1_NN.model.set_weights(player1_network.get_weights())
    player1 = Agent('tensorflow model', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)


    player2 = PyTorchAgent(pytorch_first)
    if pytorch_first:
        scores, memory, points, sp_scores = playMatches(
            player2, player1, EPISODES=episodes, logger=logger, turns_until_tau0=turns_until_tau, goes_first=1
        )

    else:
        scores, memory, points, sp_scores = playMatches(
            player1, player2, EPISODES=episodes, logger=logger, turns_until_tau0=turns_until_tau, goes_first=1
        )
    print(f"\n\n\nPytorch first: {pytorch_first}\nGames: {episodes}\nUndeterministic Turns: {turns_until_tau}")
    print(f"Scores: {scores}")


def tourney2(run_version=0, player1version=8, turns_until_tau=10, episodes=10, pytorch_first=False):
    player1 = PyTorchAgent(True)
    player2 = PyTorchAgent(False)

    scores, memory, points, sp_scores = playMatches(
        player2, player1, EPISODES=episodes, logger=logger, turns_until_tau0=turns_until_tau, goes_first=1
    )
    print(f"\n\n\nPytorch first: {pytorch_first}\nGames: {episodes}\nUndeterministic Turns: {turns_until_tau}")
    print(f"Scores: {scores}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("-pytorch_first", action='store_true', help='Pytorch model goes first, if not provided, tensorflow model goes first.')
    a.add_argument("-games", type=int, default=10, help="number of games to play")
    a.add_argument("-turns", type=int, default=10, help="number of turns until tensorflow model plays deterministically")

    args = a.parse_args()
    tourney2()
    exit(0)

    tourney(turns_until_tau=args.turns, episodes=args.games, pytorch_first=args.pytorch_first)
