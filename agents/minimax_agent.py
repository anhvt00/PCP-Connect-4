
import numpy as np
from typing import Optional, List, Tuple
from game_utils import (
    GenMove,
    PlayerAction,
    BOARD_COLS,
    BOARD_ROWS,
    INDEX_HIGHEST_ROW,
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
    check_end_state,
    GameState,
    apply_player_action,
)


def get_valid_columns(board: np.ndarray) -> List[int]:
    """
    Identify columns into which a piece may be dropped.

    Args:
        board (np.ndarray): Current 6×7 game board.

    Returns:
        List[int]: Indices of columns where the top cell is empty.
    """
    return [col for col in range(BOARD_COLS)
            if board[INDEX_HIGHEST_ROW, col] == NO_PLAYER]


def simulate_move(
    board: np.ndarray,
    column: int,
    player: np.int8
) -> np.ndarray:
    """
    Apply a move by copying the board and dropping a piece.

    Args:
        board (np.ndarray): Current board state.
        column (int): Column index to play.
        player (np.int8): Player piece (PLAYER1 or PLAYER2).

    Returns:
        np.ndarray: New board after move.

    Raises:
        ValueError: If the column is full or out of bounds.
    """
    new_board = board.copy()
    apply_player_action(new_board, PlayerAction(column), player)
    return new_board


def score_position(
    board: np.ndarray,
    player: np.int8
) -> int:
    """
    Static evaluation of board favoring center control and 4-in-a-row windows.

    Args:
        board (np.ndarray): Current board state.
        player (np.int8): Player for whom to score.

    Returns:
        int: Heuristic score where positive favors `player` and negative favors opponent.
    """
    opponent = PLAYER2 if player == PLAYER1 else PLAYER1
    score = 0
    # center column control
    center_col = board[:, BOARD_COLS // 2]
    score += int(np.count_nonzero(center_col == player)) * 3

    def evaluate_window(window: np.ndarray) -> int:
        """Score a 4-cell slice for player and opponent counts."""
        p_count = int(np.count_nonzero(window == player))
        o_count = int(np.count_nonzero(window == opponent))
        if p_count == 4:
            return 100
        if p_count == 3 and o_count == 0:
            return 5
        if p_count == 2 and o_count == 0:
            return 2
        if o_count == 4:
            return -100
        if o_count == 3 and p_count == 0:
            return -4
        if o_count == 2 and p_count == 0:
            return -2
        return 0

    # horizontal
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            score += evaluate_window(board[r, c:c+4])
    # vertical
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            score += evaluate_window(board[r:r+4, c])
    # positive diagonal (/)
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            window = np.array([board[r+i, c+3-i] for i in range(4)])
            score += evaluate_window(window)
    # negative diagonal (\)
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            window = np.array([board[r+i, c+i] for i in range(4)])
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
    Minimax search with alpha-beta pruning.

    Args:
        board (np.ndarray): Current board state.
        depth (int): Remaining search depth.
        alpha (float): Best already-explored option along path to root for maximizer.
        beta (float): Best already-explored option along path to root for minimizer.
        maximizing_player (bool): True if current move is maximizing for `player`.
        player (np.int8): Root player whose score is being optimized.

    Returns:
        Tuple[float, Optional[int]]:
            Best score and corresponding column index (None at terminal).
    """
    valid_cols = get_valid_columns(board)
    terminal = (
        depth == 0 or
        check_end_state(board, player) != GameState.STILL_PLAYING or
        not valid_cols
    )
    if terminal:
        return float(score_position(board, player)), None

    opponent = PLAYER2 if player == PLAYER1 else PLAYER1
    best_col: Optional[int] = None

    if maximizing_player:
        value = -np.inf
        for col in valid_cols:
            child = simulate_move(board, col, player)
            score, _ = minimax_ab(child, depth-1, alpha, beta, False, player)
            if score > value:
                value, best_col = score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # prune remaining branches
        return value, best_col
    else:
        value = np.inf
        for col in valid_cols:
            child = simulate_move(board, col, opponent)
            score, _ = minimax_ab(child, depth-1, alpha, beta, True, player)
            if score < value:
                value, best_col = score, col
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
    Select a move using alpha-beta–pruned Minimax of fixed depth.

    Args:
        board (np.ndarray): Current board configuration.
        player (np.int8): Player to move.
        state (Optional[object]): Unused save-state.

    Returns:
        Tuple[PlayerAction, Optional[object]]:
            Chosen column and None state.
    """
    _, best_col = minimax_ab(
        board,
        depth=5,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=player
    )
    if best_col is None:
        best_col = np.random.choice(get_valid_columns(board))
    return PlayerAction(int(best_col)), None


# Expose the agent under GenMove alias
Agent: GenMove = minimax_move


