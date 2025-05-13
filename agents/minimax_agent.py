import numpy as np
from typing import Any, List, Optional, Tuple
from game_utils import *


class MinimaxAgent:
    """Alpha-Beta pruned Minimax agent"""

    def __init__(self, depth: int = 5):
        """
        Args:
            depth (int): Maximum search depth.
        """
        self.depth = depth

    def get_valid_columns(self, board: np.ndarray) -> List[int]:
        """
        Identify all columns where a new piece can be dropped.

        Returns:
            List[int]: Column indices not yet full.
        """
        return [
            col for col in range(BOARD_COLS)
            if board[INDEX_HIGHEST_ROW, col] == NO_PLAYER
        ]

    def simulate_move(
        self,
        board: np.ndarray,
        column: int,
        player: np.int8
    ) -> np.ndarray:
        """
        Copy the board and apply a player's move.

        Args:
            board: Current game state.
            column: Column to drop into.
            player: PLAYER1 or PLAYER2.

        Returns:
            New board state after the move.
        """
        # Work on a fresh copy so original board isn't mutated
        new_board = board.copy()
        apply_player_action(new_board, PlayerAction(column), player)
        return new_board

    def score_position(
        self,
        board: np.ndarray,
        player: np.int8
    ) -> int:
        """
        Heuristic evaluation favoring center control and potential connects.

        Args:
            board: Current game state.
            player: Whose perspective to score.

        Returns:
            Integer score (higher is better for `player`).
        """
        opponent = PLAYER2 if player == PLAYER1 else PLAYER1
        score = 0

        # 1) Center column occupancy bonus
        center_col = board[:, BOARD_COLS // 2]
        score += int(np.count_nonzero(center_col == player)) * 3

        def evaluate_window(window: np.ndarray) -> int:
            """
            Score a slice of 4 cells based on counts of player vs. opponent.
            """
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

        # 2) Horizontal windows
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 3):
                window = board[r, c:c+4]
                score += evaluate_window(window)

        # 3) Vertical windows
        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 3):
                window = board[r:r+4, c]
                score += evaluate_window(window)

        # 4) Positive diagonal (/) windows
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = np.array([board[r + i, c + 3 - i] for i in range(4)])
                score += evaluate_window(window)

        # 5) Negative diagonal (\) windows
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                score += evaluate_window(window)

        return score

    def minimax_ab(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        player: np.int8
    ) -> Tuple[float, Optional[int]]:
        """
        Recursive Minimax with Alpha-Beta pruning.

        Args:
            board: Current game state.
            depth: Remaining search depth.
            alpha: Best value for maximizer so far.
            beta: Best value for minimizer so far.
            maximizing_player: True to maximize `player` score.
            player: Root player whose score we optimize.

        Returns:
            (best_score, best_column)
        """
        valid_cols = self.get_valid_columns(board)
        # Terminal if depth==0, game over, or no moves left
        is_terminal = (
            depth == 0 or
            check_end_state(board, player) != GameState.STILL_PLAYING or
            not valid_cols
        )
        if is_terminal:
            return float(self.score_position(board, player)), None

        opponent = PLAYER2 if player == PLAYER1 else PLAYER1
        best_col: Optional[int] = None

        if maximizing_player:
            value = -np.inf
            for col in valid_cols:
                # simulate move and recurse as minimizing player
                child = self.simulate_move(board, col, player)
                score, _ = self.minimax_ab(child, depth - 1, alpha, beta, False, player)
                if score > value:
                    value, best_col = score, col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # prune remaining
            return value, best_col
        else:
            value = np.inf
            for col in valid_cols:
                # simulate move for opponent, then recurse as maximizer
                child = self.simulate_move(board, col, opponent)
                score, _ = self.minimax_ab(child, depth - 1, alpha, beta, True, player)
                if score < value:
                    value, best_col = score, col
                beta = min(beta, value)
                if alpha >= beta:
                    break  # prune remaining
            return value, best_col

    def move(
        self,
        board: np.ndarray,
        player: np.int8,
        state: Optional[Any] = None
    ) -> Tuple[PlayerAction, Optional[Any]]:
        """
        GenMove interface: choose column using Minimax and return (action, state).

        If no best column found (should be rare), pick randomly.
        """
        _, best_col = self.minimax_ab(
            board,
            depth=self.depth,
            alpha=-np.inf,
            beta=np.inf,
            maximizing_player=True,
            player=player
        )
        if best_col is None:
            valid = self.get_valid_columns(board)
            best_col = int(np.random.choice(valid))
        return PlayerAction(best_col), None


# Expose the agent under the common GenMove alias
Agent: GenMove = MinimaxAgent().move
