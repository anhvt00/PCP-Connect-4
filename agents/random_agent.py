# agents/random_agent.py
import numpy as np
from game_utils import GenMove, PlayerAction, NO_PLAYER, BOARD_COLS

def random_move(board: np.ndarray, player, state=None) -> tuple[PlayerAction, None]:
    """
    A trivial agent that chooses uniformly among non-full columns.
    """
    # columns where top cell is still empty
    valid = [c for c in range(BOARD_COLS) if board[-1, c] == NO_PLAYER]
    choice = int(np.random.choice(valid))
    return PlayerAction(choice), None

# expose under the common GenMove alias
Agent: GenMove = random_move
