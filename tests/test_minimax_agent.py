# tests/test_minimax_utils.py

import numpy as np
import pytest

from agents.minimax_agent import *
from game_utils import *


def test_get_valid_columns_initial():
    """
    All columns should be valid on an empty board.
    """
    board = initialize_game_state()
    assert get_valid_columns(board) == list(range(BOARD_COLS))


def test_get_valid_columns_after_fill():
    """
    Filled columns must be excluded from valid moves.
    """
    board = initialize_game_state()
    # fill column 0 for PLAYER1 and column 6 for PLAYER2
    for _ in range(BOARD_ROWS):
        apply_player_action(board, PlayerAction(0), PLAYER1)
        apply_player_action(board, PlayerAction(6), PLAYER2)
    valid = get_valid_columns(board)
    assert 0 not in valid
    assert 6 not in valid
    assert len(valid) == BOARD_COLS - 2


def test_simulate_move_immutability():
    """
    simulate_move should not modify the original board and return a new board.
    """
    original = initialize_game_state()
    new_board = simulate_move(original, 2, PLAYER1)
    # original remains unchanged
    assert original[0, 2] == 0  # NO_PLAYER
    # new_board has the move applied
    assert new_board[0, 2] == PLAYER1
    # no shared memory
    assert not np.shares_memory(original, new_board)


def test_score_position_center_weighting():
    """
    score_position should give positive score for center control.
    """
    board = initialize_game_state()
    center = BOARD_COLS // 2
    apply_player_action(board, PlayerAction(center), PLAYER1)
    # center weighting = 3 points per disc in center column
    assert score_position(board, PLAYER1) >= 3


def test_score_position_detects_three_in_row():
    """
    score_position should detect a 3-in-a-row threat (+5).
    """
    board = initialize_game_state()
    for col in [0, 1, 2]:
        apply_player_action(board, PlayerAction(col), PLAYER1)
    # no immediate win, but a 3-in-row yields at least +5
    assert score_position(board, PLAYER1) >= 5


def test_minimax_ab_depth_zero_returns_static_score():
    """
    minimax_ab with depth=0 should return static evaluation and no move.
    """
    board = initialize_game_state()
    score, move = minimax_ab(
        board,
        depth=0,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    assert move is None
    assert isinstance(score, (int, float, np.floating, np.integer))


def test_minimax_ab_prefers_center_depth_one():
    """
    At depth=1, minimax_ab should choose the center column for best static score.
    """
    board = initialize_game_state()
    score, best_col = minimax_ab(
        board,
        depth=1,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    assert best_col == BOARD_COLS // 2
