import pytest
import numpy as np
from agents.mcts_agent import Node, MCTS, AgentMCTS
from game_utils import *

# ----- Node class tests -----

def test_node_init_invalid_turn():
    """Node __init__ should raise ValueError for invalid turn values."""
    board = initialize_game_state()
    with pytest.raises(ValueError):
        Node(None, board, turn=0)
    with pytest.raises(ValueError):
        Node(None, board, turn=3)


def test_get_moves_count_and_generate_moves():
    """get_moves_count should match BOARD_COLS and generate_moves should create correct successors."""
    board = initialize_game_state()
    node = Node(None, board, PLAYER1)
    assert node.get_moves_count() == BOARD_COLS
    moves = node.generate_moves(board, PLAYER1)
    assert len(moves) == BOARD_COLS
    # Each move should differ by exactly one drop at row 0
    for idx, mv in enumerate(moves):
        diff = np.where(mv != board)
        assert diff[0].size == 1
        r, c = diff[0][0], diff[1][0]
        assert r == 0 and c == idx


def test_is_terminal():
    """is_terminal returns False for empty board, True for winning board."""
    empty = initialize_game_state()
    node_empty = Node(None, empty, PLAYER1)
    assert not node_empty.is_terminal()
    win_board = initialize_game_state()
    for c in range(4):
        win_board[0, c] = PLAYER2
    node_win = Node(None, win_board, PLAYER2)
    assert node_win.is_terminal()

# ----- MCTS class tests -----

def test_rollout_immediate_terminal_reward():
    """rollout should return full reward for immediate win node."""
    board = initialize_game_state()
    for c in range(4):
        board[0, c] = PLAYER1
    leaf = Node(None, board, PLAYER1)
    mcts = MCTS(symbol=PLAYER1, t=0.0)
    reward = mcts.rollout(leaf)
    assert reward == pytest.approx(1.0)


def test_compute_move_zero_time_budget():
    """compute_move with zero time budget returns (-1, -1) immediately."""
    board = initialize_game_state()
    root = Node(None, board, PLAYER1)
    mcts = MCTS(symbol=PLAYER1, t=0.0)
    row, col = mcts.compute_move(root)
    assert (row, col) == (-1, -1)


def test_rollout_terminal_win():
    """rollout should return 1.0 for a terminal win state for the agent."""
    board = initialize_game_state()
    # horizontal win for PLAYER1 on bottom row
    for c in range(4):
        board[0, c] = PLAYER1
    leaf = Node(None, board, PLAYER1)
    mcts = MCTS(symbol=PLAYER1, t=0.0)
    reward = mcts.rollout(leaf)
    assert reward == pytest.approx(1.0)


def test_compute_move_zero_time_budget():
    """compute_move with zero time should immediately return (-1, -1)."""
    board = initialize_game_state()
    root = Node(None, board, PLAYER1)
    mcts = MCTS(symbol=PLAYER1, t=0.0)
    row, col = mcts.compute_move(root)
    assert (row, col) == (-1, -1)


def test_best_child_picks_most_visited():
    """best_child should return the child with the highest visit count n."""
    board = initialize_game_state()
    parent = Node(None, board, PLAYER1)
    c1 = Node(parent, board.copy(), PLAYER2)
    c2 = Node(parent, board.copy(), PLAYER2)
    c1.n, c2.n = 1, 5
    parent.children = [c1, c2]
    mcts = MCTS(symbol=PLAYER1, t=0.0)
    best = mcts.best_child(parent)
    assert best is c2


def test_backpropagate_updates_stats():
    """backpropagate should increment n and add rewards along the path."""
    board = initialize_game_state()
    root = Node(None, board, PLAYER1)
    leaf = Node(root, board.copy(), PLAYER2)
    root.children = [leaf]
    mcts = MCTS(symbol=PLAYER2, t=0.0)
    mcts.backpropagate(leaf, 1.0)
    # both leaf and root visited once
    assert leaf.n == 1 and root.n == 1
    # reward added for correct perspective
    assert leaf.q == pytest.approx(1.0)
    assert root.q == pytest.approx(0.0)  # no reward for opponent node
