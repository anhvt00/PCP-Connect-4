import numpy as np
from typing import Optional, Tuple, Any
from game_utils import (
    GenMove,
    PlayerAction,
    BOARD_COLS,
    NO_PLAYER
)


def random_move(
    board: np.ndarray,
    player: int,
    state: Optional[Any] = None
) -> Tuple[PlayerAction, None]:
    """
    Choose a random valid column to play.

    Args:
        board (np.ndarray): Current 6×7 game board.
        player (int): Player ID (PLAYER1 or PLAYER2), unused by this agent.
        state (Optional[Any]): Unused placeholder for stateful agents.

    Returns:
        Tuple[PlayerAction, None]: The chosen column wrapped in PlayerAction, and None.
    """
    # Identify all non-full columns
    valid_columns = [c for c in range(BOARD_COLS) if board[-1, c] == NO_PLAYER]
    # Choose one at random
    choice = int(np.random.choice(valid_columns))
    return PlayerAction(choice), None


# Expose under the common GenMove alias
Agent: GenMove = random_move