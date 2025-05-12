from typing import Callable, Optional, Any
from enum import Enum
import numpy as np

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input does not have the correct type (PlayerAction).'
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros(BOARD_SHAPE, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    # Top border
    border = "|" + "=" * (BOARD_COLS * 2) + "|"
    lines = [border]
    # Rows from top (highest index) to bottom (row 0)
    for r in range(BOARD_ROWS - 1, -1, -1):
        row_pieces = []
        for c in range(BOARD_COLS):
            val = board[r, c]
            if val == PLAYER1:
                row_pieces.append(PLAYER1_PRINT)
            elif val == PLAYER2:
                row_pieces.append(PLAYER2_PRINT)
            else:
                row_pieces.append(NO_PLAYER_PRINT)
        # each cell: symbol + space
        lines.append("|" + "".join(f"{p} " for p in row_pieces) + "|")
    # Bottom border and indices
    lines.append(border)
    lines.append("|" + "".join(f"{i} " for i in range(BOARD_COLS)) + "|")
    return "\n".join(lines)


def s_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    lines = pp_board.splitlines()
    # rows are lines[1] through lines[BOARD_ROWS]
    board_lines = lines[1:1 + BOARD_ROWS]
    board = initialize_game_state()
    for i, line in enumerate(board_lines):
        content = line[1:-1]
        for j in range(BOARD_COLS):
            symbol = content[2 * j]
            if symbol == PLAYER1_PRINT:
                board[BOARD_ROWS - 1 - i, j] = PLAYER1
            elif symbol == PLAYER2_PRINT:
                board[BOARD_ROWS - 1 - i, j] = PLAYER2
            else:
                board[BOARD_ROWS - 1 - i, j] = NO_PLAYER
    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input 
    board should be modified in place, such that it's not necessary to return 
    something.
    """
    col = int(action)
    for row in range(BOARD_ROWS):
        if board[row, col] == NO_PLAYER:
            board[row, col] = player
            return
    raise ValueError(MoveStatus.FULL_COLUMN.value)


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    # Horizontal
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    # Vertical
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    # Diagonal up-right
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    # Diagonal up-left
    for r in range(BOARD_ROWS - 3):
        for c in range(3, BOARD_COLS):
            if all(board[r + i, c - i] == player for i in range(4)):
                return True
    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    elif np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is accepted as a valid move 
    or not, and if not, why.
    The provided column must be of the correct type (PlayerAction).
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.
    """
    # Type check
    if not isinstance(column, (int, np.integer)):
        return MoveStatus.WRONG_TYPE
    col = int(column)
    # Bounds check
    if col < 0 or col >= BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS
    # Full column check
    if board[INDEX_HIGHEST_ROW, col] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN
    return MoveStatus.IS_VALID


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    lines = pp_board.splitlines()
    # Extract only the BOARD_ROWS rows (skip the top border and bottom border + index line)
    board_lines = lines[1 : 1 + BOARD_ROWS]
    board = initialize_game_state()

    for i, line in enumerate(board_lines):
        # Strip the leading and trailing '|' from each line
        content = line[1:-1]  
        # Each cell is represented by a symbol and a space, e.g. "X ", "O ", or "  "
        for j in range(BOARD_COLS):
            symbol = content[2 * j]
            if symbol == PLAYER1_PRINT:
                board[BOARD_ROWS - 1 - i, j] = PLAYER1
            elif symbol == PLAYER2_PRINT:
                board[BOARD_ROWS - 1 - i, j] = PLAYER2
            else:
                board[BOARD_ROWS - 1 - i, j] = NO_PLAYER

    return board
