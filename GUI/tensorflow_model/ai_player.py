import numpy as np
from .game import GameState
from .model import Residual_CNN_tflite

from .agent import Agent

from .config import *


def setup_ai(player_version=32):
    run_version = 0

    player_NN = Residual_CNN_tflite()
    player_NN.read("connect4", run_version, player_version)
    agent = Agent('player1', 42, 42, MCTS_SIMS, CPUCT, player_NN)
    agent.mcts = None
    return agent


def convert_game_state(state):
    """
    Convert the 2D game state from the GUI code to a 1D game state for the model.
    """

    mapping = {
        'X': 1,
        'O': -1,
        'E': 0
    }

    for col in state:
        while len(col) < 6:
            col.append('E') # for empty

    # print(state)

    res = []

    for i in range(6):
        for col in state:
            element = col.pop()
            res.append(mapping[element])

    return res


ai_agent = setup_ai()

def model_column(state):
    state = convert_game_state(state)
    game_state = GameState(np.array(state), -1)
    res = ai_agent.act(game_state, 0)
    return res[0] % 7

