# tests/test_random_agent.py
import numpy as np
from game_utils import initialize_game_state, PLAYER1, NO_PLAYER, BOARD_COLS
from agents.random_agent import random_move

def test_random_agent_only_valid_columns():
    b = initialize_game_state()
    # fill columns 0 & 1 completely
    for _ in range(6):
        b[:,0] = PLAYER1
        b[:,1] = PLAYER1
    move, state = random_move(b, PLAYER1)
    assert isinstance(move, np.integer)
    assert 0 <= move < BOARD_COLS
    assert b[-1, move] == NO_PLAYER
    assert state is None
