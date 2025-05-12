# agents/minimax_agent.py

import numpy as np
from typing import Optional, List, Tuple
from game_utils import (
    GenMove,
    PlayerAction,
    NO_PLAYER,
    BOARD_COLS,
    BOARD_ROWS,
    INDEX_HIGHEST_ROW,
    apply_player_action,
    check_end_state,
    GameState,
    PLAYER1,
    PLAYER2,
)


def get_valid_columns(board: np.ndarray) -> List[int]:
    """
    Return a list of columns where a move can be made (top cell is empty).
    """
    return [col for col in range(BOARD_COLS) if board[INDEX_HIGHEST_ROW, col] == NO_PLAYER]


def simulate_move(board: np.ndarray, column: int, player: np.int8) -> np.ndarray:
    """
    Return a copy of `board` with `player` playing into `column`.
    """
    new_board = board.copy()
    apply_player_action(new_board, PlayerAction(column), player)
    return new_board


def score_position(board: np.ndarray, player: np.int8) -> int:
    """
    Enhanced evaluation:
    - Center column control
    - Score 4-cell windows based on # of player and opponent pieces
    """
    opponent = PLAYER2 if player == PLAYER1 else PLAYER1
    score = 0

    # 1) Center column control
    center_array = board[:, BOARD_COLS // 2]
    center_count = int(np.count_nonzero(center_array == player))
    score += center_count * 3

    # helper to evaluate a window of 4 cells
    def evaluate_window(window: np.ndarray) -> int:
        p_count = int(np.count_nonzero(window == player))
        o_count = int(np.count_nonzero(window == opponent))
        if p_count == 4:
            return 100
        elif p_count == 3 and o_count == 0:
            return 5
        elif p_count == 2 and o_count == 0:
            return 2
        if o_count == 4:
            return -100
        elif o_count == 3 and p_count == 0:
            return -4
        elif o_count == 2 and p_count == 0:
            return -2
        return 0

    # 2) Horizontal windows
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            window = board[r, c : c+4]
            score += evaluate_window(window)

    # 3) Vertical windows
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            window = board[r : r+4, c]
            score += evaluate_window(window)

    # 4) Positive diagonal (/)
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            window = np.array([board[r + i, c + 3 - i] for i in range(4)])
            score += evaluate_window(window)

    # 5) Negative diagonal (\)
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            window = np.array([board[r + i, c + i] for i in range(4)])
            score += evaluate_window(window)

    return score


def minimax_ab(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    player: np.int8
) -> Tuple[float, Optional[int]]:
    """
    Minimax search with Alpha-Beta pruning. Returns (score, best_column).
    """
    valid_cols = get_valid_columns(board)
    terminal = (depth == 0 or
                check_end_state(board, player) != GameState.STILL_PLAYING or
                not valid_cols)
    if terminal:
        return float(score_position(board, player)), None

    opponent = PLAYER2 if player == PLAYER1 else PLAYER1

    if maximizing_player:
        value, best_col = -np.inf, valid_cols[0]
        for col in valid_cols:
            child = simulate_move(board, col, player)
            new_score, _ = minimax_ab(child, depth-1, alpha, beta, False, player)
            if new_score > value:
                value, best_col = new_score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_col
    else:
        value, best_col = np.inf, valid_cols[0]
        for col in valid_cols:
            child = simulate_move(board, col, opponent)
            new_score, _ = minimax_ab(child, depth-1, alpha, beta, True, player)
            if new_score < value:
                value, best_col = new_score, col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_col


def minimax_move(
    board: np.ndarray,
    player: np.int8,
    state: Optional[object] = None
) -> Tuple[PlayerAction, Optional[object]]:
    """
    Select a move using alpha-beta–pruned Minimax with fixed depth.
    """
    score, best_col = minimax_ab(
        board,
        depth=5,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=player
    )
    if best_col is None:
        best_col = int(np.random.choice(get_valid_columns(board)))
    return PlayerAction(best_col), None


# Expose the agent under GenMove alias
Agent: GenMove = minimax_move


def string_to_board(board_str: str) -> np.ndarray:
    """
    Convert a string representation of the board into a numpy array.
    """
    rows = board_str.strip().split("\n")
    board = np.array([[int(cell) for cell in row.split()] for row in rows], dtype=np.int8)
    return board
