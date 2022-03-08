import argparse

from game import Game
from funcs import playMatchesBetweenVersions
import logging

logger = logging.getLogger("foo")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="play connect4")
parser.add_argument('-p1', type=int, default=-1, required=False,
                    help="Integer for player 1.  -1 is human, nonnegative is a model.")
parser.add_argument('-p2', type=int, default=8, required=False,
                    help="Integer for player 2.  -1 is human, nonnegative is a model.")

args = parser.parse_args()

env = Game()
playMatchesBetweenVersions(
    env=env,
    run_version=0,
    player1version=args.p1,
    player2version=args.p2,
    EPISODES=1,
    logger=logger,
    turns_until_tau0=0
)

