import time
import numpy as np
import random
import math
from typing import Optional, List, Tuple, Any
from game_utils import *

class Node:
    """Monte Carlo tree node storing the game state and MCTS statistics."""
    def __init__(self, parent: Optional['Node'], board: np.ndarray, turn: int) -> None:
        """
        Initialize a Node.

        Args:
            parent (Optional[Node]): Parent node in the tree, or None for root.
            board (np.ndarray): Current board state (6×7 array).
            turn (int): Player who just moved (PLAYER1 or PLAYER2).

        """
        self.parent = parent
        self.board = board.copy()  # snapshot for isolation
        self.turn = turn  # last mover
        self.q = 0.0      # total reward from simulations
        self.n = 0        # visit count
        self.children: List['Node'] = []
        self.expanded = False  # indicates if all children have been generated

    def is_terminal(self) -> bool:
        """
        Check if this node represents a terminal state (win or draw).

        Returns:
            bool: True if no further moves or a win condition is met.
        """
        return check_end_state(self.board, self.turn) != GameState.STILL_PLAYING

    def get_moves_count(self) -> int:
        """
        Count legal moves from this state.

        Returns:
            int: Number of valid columns where a piece can be dropped.
        """
        return sum(
            check_move_status(self.board, np.int8(c)) == MoveStatus.IS_VALID
            for c in range(BOARD_COLS)
        )

    def generate_moves(self, board: np.ndarray, turn: int) -> List[np.ndarray]:
        """
        Generate all successor board states for a move by `turn`.

        Args:
            board (np.ndarray): Current board.
            turn (int): Player to move (PLAYER1 or PLAYER2).

        Returns:
            List[np.ndarray]: List of new board arrays after each legal play.
        """
        moves = []
        for c in range(BOARD_COLS):
            if board[5, c] != 0:
                continue  # column full
            for r in range(6):
                if board[r, c] == 0:
                    new = board.copy()
                    new[r, c] = turn
                    moves.append(new)
                    break  # move to next column
        return moves

    def add_child(self) -> None:
        """
        Expand one unexplored child node, if available.
        Marks `expanded = True` when no further children can be added.
        """
        if self.expanded or self.is_terminal():
            return
        next_player = PLAYER1 if self.turn == PLAYER2 else PLAYER2
        seen = {child.board.tobytes() for child in self.children}
        for c in range(BOARD_COLS):
            if self.board[5, c] != 0:
                continue
            for r in range(6):
                if self.board[r, c] == 0:
                    new_board = self.board.copy()
                    new_board[r, c] = next_player
                    key = new_board.tobytes()
                    if key not in seen:
                        self.children.append(Node(self, new_board, next_player))
                        return
                    break
        self.expanded = True

class MCTS:
    """Monte Carlo Tree Search agent using UCT and random simulations."""

    def __init__(self, symbol: int, t: float) -> None:
        """
        Initialize MCTS agent.

        Args:
            symbol (int): The agent's player ID (PLAYER1 or PLAYER2).
            t (float): Time budget in seconds per move.

        Raises:
            ValueError: If `symbol` is invalid.
        """
        if symbol not in (PLAYER1, PLAYER2):
            raise ValueError(f"Invalid symbol for MCTS agent: {symbol}")
        self.symbol = symbol
        self.t = t

    def compute_move(self, node: "Node") -> Tuple[int, int]:
        """
        Run MCTS for up to `t` seconds and select a move.

        Args:
            node (Node): Root node of current state.

        Returns:
            Tuple[int, int]: (row, col) of chosen move, or (-1, -1).
        """
        time0 = time.time()
        while (time.time() - time0) < self.t:
            leaf = self.select(node)
            if leaf is None:
                return (-1, -1)
            result = self.rollout(leaf)
            self.backpropagate(leaf, result)
        selected = self.best_child(node)
        if selected is None:
            return (-1, -1)
        diff = np.where(selected.board != node.board)
        if diff[0].size:
            return int(diff[0][0]), int(diff[1][0])
        return (-1, -1)

    def move(self, board: np.ndarray, player: int, state: Any = None) -> Tuple[PlayerAction, Any]:
        """
        GenMove interface: wrap compute_move.

        Args:
            board (np.ndarray): Current board.
            player (int): Player to move.
            state (Any): Unused placeholder.

        Returns:
            Tuple[PlayerAction, Any]: Column action and unchanged state.
        """
        root = Node(None, board, player)
        _, col = self.compute_move(root)
        return PlayerAction(col), state

    def select(self, node: "Node") -> Optional["Node"]:
        """
        Traverse tree via UCT until a leaf is found, expanding if needed.
        """
        while self.fully_expanded(node):
            tmp = self.select_uct(node)
            if tmp is node:
                break
            node = tmp
        if node.is_terminal():
            return node
        node.add_child()
        for child in node.children:
            if child.n == 0:
                return child
        return node

    def select_uct(self, node: "Node") -> "Node":
        """
        Select child maximizing UCT score: Q/N + exploration.
        """
        best_score = -math.inf
        best_node = node
        for child in node.children:
            if child.n == 0:
                score = math.inf
            else:
                exploitation = child.q / child.n
                exploration = 2 * math.sqrt(math.log(node.n) / child.n)
                score = exploitation + exploration
            if score > best_score:
                best_score, best_node = score, child
        return best_node

    def fully_expanded(self, node: "Node") -> bool:
        """
        Check if all possible children have been generated and visited.
        """
        possible = node.get_moves_count()
        if len(node.children) < possible:
            return False
        return all(child.n > 0 for child in node.children)

    def rollout(self, node: "Node") -> float:
        """
        Simulate random playout until terminal state, returning reward.
        """
        board = node.board.copy()
        turn = node.turn
        while True:
            state = check_end_state(board, turn)
            if state == GameState.IS_WIN:
                return 1.0 if turn == self.symbol else 0.0
            if state == GameState.IS_DRAW:
                return 0.5
            moves = node.generate_moves(board, turn)
            if not moves:
                return 0.5
            board = random.choice(moves)
            turn = PLAYER1 if turn == PLAYER2 else PLAYER2

    def backpropagate(self, node: "Node", result: float) -> None:
        """
        Propagate simulation result up to the root, updating Q and N.
        """
        while node:
            node.n += 1
            node.q += result if node.turn == self.symbol else (1.0 - result)
            node = node.parent

    def best_child(self, node: "Node") -> Optional["Node"]:
        """
        Select the child with the highest visit count.
        """
        if not node.children:
            return None
        return max(node.children, key=lambda c: c.n)

# Expose agent
AgentMCTS: GenMove = MCTS(symbol=PLAYER1, t=5.0).move