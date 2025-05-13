import numpy as np
import pytest

from agents.minimax_agent import MinimaxAgent
from game_utils import *

def test_minimax_ab_returns_static_eval_at_depth_zero():
    """At depth=0, minimax_ab should return the heuristic value and no move."""
    board = initialize_game_state()
    agent = MinimaxAgent(depth=4)
    static_score, best_col = agent.minimax_ab(
        board,
        depth=0,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    expected = float(agent.score_position(board, PLAYER1))
    assert static_score == expected
    assert best_col is None

def test_minimax_ab_prefers_center_on_empty_depth_one():
    """
    At depth=1 on an empty board, minimax_ab should choose the center column
    because the static evaluator gives it a +3 bonus.
    """
    board = initialize_game_state()
    agent = MinimaxAgent(depth=1)
    _, best_col = agent.minimax_ab(
        board,
        depth=1,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    assert best_col == BOARD_COLS // 2

def test_minimax_ab_finds_immediate_win_depth_one():
    """
    With three in a row for PLAYER1, minimax_ab(depth=1) should pick the winning move.
    """
    board = initialize_game_state()
    # place three horizontally at row 0, cols 0–2
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER1)
    agent = MinimaxAgent(depth=1)
    _, best_col = agent.minimax_ab(
        board,
        depth=1,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    assert best_col == 3
    # verify that move indeed yields a win
    new_board = agent.simulate_move(board, best_col, PLAYER1)
    assert check_end_state(new_board, PLAYER1) == GameState.IS_WIN


def test_score_position_two_in_a_row_horizontal():
    """Two in a row horizontally should score +2 for that player."""
    board = initialize_game_state()
    # place two PLAYER1 pieces at (0,0) and (0,1)
    apply_player_action(board, PlayerAction(0), PLAYER1)
    apply_player_action(board, PlayerAction(1), PLAYER1)
    agent = MinimaxAgent(depth=1)
    score = agent.score_position(board, PLAYER1)
    # expect a window with 2 pieces, no opponents → +2
    assert score == 2

def test_score_position_block_three_opponent():
    """Three in a row by opponent should score -4 for player."""
    board = initialize_game_state()
    # place three PLAYER2 pieces at (0,0),(0,1),(0,2)
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER2)
    agent = MinimaxAgent(depth=1)
    score = agent.score_position(board, PLAYER1)
    # opponent has a window of 3 → each window gives -4
    assert score <= -4

def test_minimax_ab_blocks_opponent_win_at_depth_two():
    """
    With opponent threatening a horizontal win in 3, depth=2 should block it.
    """
    board = initialize_game_state()
    # opponent (PLAYER2) has three at cols 0-2
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER2)
    agent = MinimaxAgent(depth=2)
    _, best_col = agent.minimax_ab(
        board,
        depth=2,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    # best move is to block at column 3
    assert best_col == 3
    # verify block prevents immediate win
    blocked = agent.simulate_move(board, best_col, PLAYER1)
    assert check_end_state(blocked, PLAYER2) != GameState.IS_WIN


def test_score_position_vertical_two_in_a_row():
    """Two in a row vertically (in the center) should score 8 (6 center + 2 window)."""
    board = initialize_game_state()
    # Drop two pieces into the center column (index 3)
    apply_player_action(board, PlayerAction(3), PLAYER1)
    apply_player_action(board, PlayerAction(3), PLAYER1)
    agent = MinimaxAgent(depth=1)
    score = agent.score_position(board, PLAYER1)
    assert score == 8  # 2*3 center bonus + 1*2 for the vertical window

def test_minimax_ab_prefers_diagonal_setup_depth_one():
    """
    Given a backwardslash diagonal setup but using only depth=1 static eval,
    minimax_ab will choose column 1 (maximizing the center/3-window bonuses).
    """
    board = initialize_game_state()
    # Build a \ diagonal of PLAYER1 (0,0),(1,1),(2,2) with blockers
    apply_player_action(board, PlayerAction(0), PLAYER1)
    apply_player_action(board, PlayerAction(1), PLAYER2)
    apply_player_action(board, PlayerAction(1), PLAYER1)
    apply_player_action(board, PlayerAction(2), PLAYER2)
    apply_player_action(board, PlayerAction(2), PLAYER2)
    apply_player_action(board, PlayerAction(2), PLAYER1)

    agent = MinimaxAgent(depth=1)
    _, best_col = agent.minimax_ab(
        board,
        depth=1,
        alpha=-np.inf,
        beta=np.inf,
        maximizing_player=True,
        player=PLAYER1
    )
    assert best_col == 1
