# tests/test_game_utils.py

import numpy as np
import pytest

from game_utils import *
# -----------------------------------------------------------------------------
# 3.2 initialize_game_state() tests
# -----------------------------------------------------------------------------
def test_initialize_game_state_creates_empty_board():
    """
    Test that initialize_game_state returns a NumPy array of shape BOARD_SHAPE,
    dtype BoardPiece, filled entirely with NO_PLAYER.
    """
    board = initialize_game_state()
    assert isinstance(board, np.ndarray), "Should return a NumPy ndarray"
    assert board.shape == BOARD_SHAPE, f"Expected shape {BOARD_SHAPE}"
    assert board.dtype == BoardPiece, "Expected dtype to be BoardPiece (np.int8)"
    assert np.all(board == NO_PLAYER), "All entries should equal NO_PLAYER (0)"


# -----------------------------------------------------------------------------
# 3.3 pretty_print_board() tests
# -----------------------------------------------------------------------------
def test_pretty_print_empty_board():
    """
    Test that an empty board is printed with borders, blank rows,
    and a bottom index line.
    """
    board = initialize_game_state()
    s = pretty_print_board(board)
    lines = s.splitlines()

    # Top and bottom border strings
    border = "|" + "=" * (BOARD_COLS * 2) + "|"
    # One empty row: '|' + two spaces per column + '|'
    empty_row = "|" + "  " * BOARD_COLS + "|"
    # Column indices line
    index_line = "|" + "".join(f"{i} " for i in range(BOARD_COLS)) + "|"

    # Check border, empty rows, bottom border, and index line
    assert lines[0] == border
    assert lines[1:1 + BOARD_ROWS] == [empty_row] * BOARD_ROWS
    assert lines[1 + BOARD_ROWS] == border
    assert lines[2 + BOARD_ROWS] == index_line


def test_pretty_print_board_with_pieces():
    """
    Test that pretty_print_board correctly places 'O' and 'X'
    for PLAYER2 and PLAYER1, respectively.
    """
    board = initialize_game_state()
    # Place O at (row=0, col=0) and X at (row=1, col=3)
    board[0, 0] = PLAYER2
    board[1, 3] = PLAYER1

    lines = pretty_print_board(board).splitlines()

    # Bottom row is printed at line index = 1 + (BOARD_ROWS - 1 - 0) = BOARD_ROWS
    expected_bottom = "|" + "O " + "  " * (BOARD_COLS - 1) + "|"
    assert lines[BOARD_ROWS] == expected_bottom, "O should appear at bottom-left"

    # Next row up is printed at line index = BOARD_ROWS - 1
    expected_second = "|" + "  " * 3 + "X " + "  " * (BOARD_COLS - 4) + "|"
    assert lines[BOARD_ROWS - 1] == expected_second, "X should appear at (row=1, col=3)"


# -----------------------------------------------------------------------------
# 3.4 string_to_board() round-trip test
# -----------------------------------------------------------------------------
def test_string_to_board_roundtrip():
    """
    Ensure that pretty_print_board → string_to_board is lossless.
    """
    b1 = initialize_game_state()
    # Scatter a few pieces at various positions
    b1[0, 2] = PLAYER1
    b1[0, 3] = PLAYER2
    b1[2, 5] = PLAYER1

    s = pretty_print_board(b1)
    b2 = string_to_board(s)

    assert np.array_equal(b1, b2), "Round-trip conversion failed"


# -----------------------------------------------------------------------------
# 3.5 apply_player_action() tests
# -----------------------------------------------------------------------------
def test_apply_player_action_stacks_correctly():
    """
    Verify pieces drop to the lowest available row in the specified column.
    """
    b = initialize_game_state()
    apply_player_action(b, PlayerAction(4), PLAYER1)
    assert b[0, 4] == PLAYER1, "First piece should land at row 0"
    apply_player_action(b, PlayerAction(4), PLAYER2)
    assert b[1, 4] == PLAYER2, "Second piece should stack on top (row 1)"


def test_apply_player_action_full_column_raises():
    """
    Confirm that applying an action to a full column raises ValueError.
    """
    b = initialize_game_state()
    # Fill column 0
    for _ in range(BOARD_ROWS):
        apply_player_action(b, PlayerAction(0), PLAYER1)
    with pytest.raises(ValueError):
        apply_player_action(b, PlayerAction(0), PLAYER2)


# -----------------------------------------------------------------------------
# 3.6 connected_four() tests
# -----------------------------------------------------------------------------
def test_connected_four_horizontal():
    """
    Horizontal line of 4 should be detected as a win.
    """
    b = initialize_game_state()
    for c in range(4):
        b[0, c] = PLAYER1
    assert connected_four(b, PLAYER1)


def test_connected_four_vertical():
    """
    Vertical line of 4 should be detected as a win.
    """
    b = initialize_game_state()
    for r in range(4):
        b[r, 0] = PLAYER2
    assert connected_four(b, PLAYER2)


def test_connected_four_diagonals():
    """
    Both diagonal directions (\ and /) should be detected.
    """
    b = initialize_game_state()
    # up-right diagonal
    for i in range(4):
        b[i, i] = PLAYER1
    # up-left diagonal
    for i in range(4):
        b[i, 3 - i] = PLAYER2
    assert connected_four(b, PLAYER1)
    assert connected_four(b, PLAYER2)


def test_connected_four_negative():
    """
    Less than 4 in a row must not count as a win.
    """
    b = initialize_game_state()
    b[0, 0:3] = PLAYER1
    assert not connected_four(b, PLAYER1)


# -----------------------------------------------------------------------------
# 3.7 check_end_state() tests
# -----------------------------------------------------------------------------
def test_check_end_state_win():
    """
    If connected_four is True, state should be IS_WIN.
    """
    b = initialize_game_state()
    for c in range(4):
        b[0, c] = PLAYER1
    assert check_end_state(b, PLAYER1) == GameState.IS_WIN


def test_check_end_state_draw():
    """
    A full board with no four‐in‐a‐row for either player should be IS_DRAW.
    We construct a 6×7 pattern with at most 2 identical pieces in any direction.
    """
    import numpy as np
    from game_utils import BoardPiece, PLAYER1, PLAYER2, connected_four, check_end_state, GameState

    # A safe “no‐win” pattern: runs of at most 2, no diagonal runs of 4
    b = np.array([
        [1, 2, 2, 1, 1, 2, 2],
        [2, 1, 1, 2, 2, 1, 1],
        [1, 2, 2, 1, 1, 2, 2],
        [2, 1, 1, 2, 2, 1, 1],
        [1, 2, 2, 1, 1, 2, 2],
        [2, 1, 1, 2, 2, 1, 1],
    ], dtype=BoardPiece)

    # Verify no wins for either player
    assert not connected_four(b, PLAYER1), "Pattern must not contain any 4‐in‐a‐row for PLAYER1"
    assert not connected_four(b, PLAYER2), "Pattern must not contain any 4‐in‐a‐row for PLAYER2"

    # Now the game state should be IS_DRAW for both
    assert check_end_state(b, PLAYER1) == GameState.IS_DRAW
    assert check_end_state(b, PLAYER2) == GameState.IS_DRAW

def test_check_end_state_still_playing():
    """
    With empty spaces and no win, state should be STILL_PLAYING.
    """
    b = initialize_game_state()
    b[0, 0] = PLAYER1
    assert check_end_state(b, PLAYER1) == GameState.STILL_PLAYING


# -----------------------------------------------------------------------------
# 3.8 check_move_status() tests
# -----------------------------------------------------------------------------
def test_check_move_status_valid():
    """
    A valid move (correct type, in bounds, non-full) returns IS_VALID.
    """
    b = initialize_game_state()
    assert check_move_status(b, PlayerAction(3)) == MoveStatus.IS_VALID


def test_check_move_status_wrong_type():
    """
    Non-integer column inputs should return WRONG_TYPE.
    """
    b = initialize_game_state()
    assert check_move_status(b, 3.14) == MoveStatus.WRONG_TYPE


def test_check_move_status_out_of_bounds():
    """
    Columns <0 or >= BOARD_COLS should return OUT_OF_BOUNDS.
    """
    b = initialize_game_state()
    assert check_move_status(b, PlayerAction(-1)) == MoveStatus.OUT_OF_BOUNDS
    assert check_move_status(b, PlayerAction(BOARD_COLS)) == MoveStatus.OUT_OF_BOUNDS


def test_check_move_status_full_column():
    """
    A move in a full column should return FULL_COLUMN.
    """
    b = initialize_game_state()
    for _ in range(BOARD_ROWS):
        apply_player_action(b, PlayerAction(2), PLAYER1)
    assert check_move_status(b, PlayerAction(2)) == MoveStatus.FULL_COLUMN
