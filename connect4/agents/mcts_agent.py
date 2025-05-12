# agents/mcts_agent.py

import numpy as np
import math
from typing import Optional, Dict, List, Tuple
from game_utils import (
    GenMove,
    PlayerAction,
    NO_PLAYER,
    BOARD_COLS,
    apply_player_action,
    check_move_status,
    check_end_state,
    GameState,
    MoveStatus,
    PLAYER1,
    PLAYER2,
)
import random
from agents.minimax_agent import score_position  # static evaluation for hybrid rollouts

class MCTSNode:
    def __init__(
        self,
        board: np.ndarray,
        parent: Optional['MCTSNode'],
        player: np.int8
    ):
        """
        node.player is the player who just moved to reach this node.
        """
        self.board = board.copy()
        self.parent = parent
        self.player = player
        self.children: Dict[int, MCTSNode] = {}
        self.wins = 0.0
        self.visits = 0

    def valid_moves(self) -> List[int]:
        """Return list of valid column indices for next move."""
        return [c for c in range(BOARD_COLS)
                if check_move_status(self.board, np.int8(c)) == MoveStatus.IS_VALID]

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.valid_moves())

    def uct_score(self, total_visits: int, c: float) -> float:
        """
        Upper Confidence Bound for Trees (UCT) score.
        """
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits
                + c * math.sqrt(math.log(total_visits) / self.visits))

class MCTSAgent:
    """
    Monte Carlo Tree Search agent with UCT, hybrid static rollouts, and move ordering.
    """
    def __init__(self, iterations: int = 2000, c: float = 1.4, rollout_depth: int = 6):
        self.iterations = iterations
        self.c = c
        self.rollout_depth = rollout_depth

    def move(
        self,
        board: np.ndarray,
        player: np.int8,
        state: Optional[object] = None
    ) -> Tuple[PlayerAction, Optional[object]]:
        # Root node: previous mover is opponent
        root = MCTSNode(
            board,
            parent=None,
            player=PLAYER2 if player == PLAYER1 else PLAYER1
        )

        for _ in range(self.iterations):
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # pick the child with highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return PlayerAction(best_move), None

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Traverse the tree using UCT until a leaf, expand if possible.
        """
        while node.children:
            total = sum(child.visits for child in node.children.values())
            # order children by UCT score
            node = max(
                node.children.values(),
                key=lambda n: n.uct_score(total, self.c)
            )
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Add one new child by exploring an untried move, in center-first order.
        """
        # prioritize center columns for expansion
        moves = sorted(node.valid_moves(), key=lambda c: abs(c - BOARD_COLS//2))
        for move in moves:
            if move not in node.children:
                new_board = node.board.copy()
                next_player = PLAYER2 if node.player == PLAYER1 else PLAYER1
                apply_player_action(new_board, PlayerAction(move), next_player)
                child = MCTSNode(new_board, parent=node, player=next_player)
                node.children[move] = child
                return child
        return node

    def _simulate(self, node: MCTSNode) -> float:
        """
        Perform a hybrid playout: heuristic moves up to rollout_depth, then static eval.
        Returns 1.0 if node.player eventual win, 0.0 if loss, 0.5 draw.
        """
        b = node.board.copy()
        current = PLAYER2 if node.player == PLAYER1 else PLAYER1
        for depth in range(self.rollout_depth):
            state = check_end_state(b, current)
            if state == GameState.IS_WIN:
                return 1.0 if current == node.player else 0.0
            if state == GameState.IS_DRAW:
                return 0.5
            # heuristic move: win, block, center, random
            # recompute valid moves for current board state
            valid = [
                c for c in range(BOARD_COLS)
                if check_move_status(b, np.int8(c)) == MoveStatus.IS_VALID
            ]
            move = self._heuristic_move(b, current, valid)
            apply_player_action(b, PlayerAction(move), current)
            current = PLAYER2 if current == PLAYER1 else PLAYER1
        # after rollout_depth, use static evaluation
        static = score_position(b, node.player)
        if static > 0:
            return 1.0
        if static < 0:
            return 0.0
        return 0.5
        static = score_position(b, node.player)
        if static > 0:
            return 1.0
        if static < 0:
            return 0.0
        return 0.5

    def _heuristic_move(self, board: np.ndarray, current: np.int8, valid: List[int]) -> int:
        """
        Heuristic move: win, block, center, then random.
        """
        # winning move
        for c in valid:
            temp = board.copy()
            apply_player_action(temp, PlayerAction(c), current)
            if check_end_state(temp, current) == GameState.IS_WIN:
                return c
        # block opponent
        opponent = PLAYER2 if current == PLAYER1 else PLAYER1
        for c in valid:
            temp = board.copy()
            apply_player_action(temp, PlayerAction(c), opponent)
            if check_end_state(temp, opponent) == GameState.IS_WIN:
                return c
        # center
        center = BOARD_COLS // 2
        if center in valid:
            return center
        # random
        return random.choice(valid)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Propagate simulation result up to the root.
        """
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent

# Expose agent
AgentMCTS: GenMove = MCTSAgent().move
# agents/mcts_agent.py

import numpy as np
import math
from typing import Optional, Dict, List, Tuple
from game_utils import (
    GenMove,
    PlayerAction,
    NO_PLAYER,
    BOARD_COLS,
    apply_player_action,
    check_move_status,
    check_end_state,
    GameState,
    MoveStatus,
    PLAYER1,
    PLAYER2,
)
import random
from agents.minimax_agent import score_position  # static evaluation for hybrid rollouts

class MCTSNode:
    def __init__(
        self,
        board: np.ndarray,
        parent: Optional['MCTSNode'],
        player: np.int8
    ):
        """
        node.player is the player who just moved to reach this node.
        """
        self.board = board.copy()
        self.parent = parent
        self.player = player
        self.children: Dict[int, MCTSNode] = {}
        self.wins = 0.0
        self.visits = 0

    def valid_moves(self) -> List[int]:
        """Return list of valid column indices for next move."""
        return [c for c in range(BOARD_COLS)
                if check_move_status(self.board, np.int8(c)) == MoveStatus.IS_VALID]

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.valid_moves())

    def uct_score(self, total_visits: int, c: float) -> float:
        """
        Upper Confidence Bound for Trees (UCT) score.
        """
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits
                + c * math.sqrt(math.log(total_visits) / self.visits))

class MCTSAgent:
    """
    Monte Carlo Tree Search agent with UCT, hybrid static rollouts, and move ordering.
    """
    def __init__(self, iterations: int = 2000, c: float = 1.4, rollout_depth: int = 6):
        self.iterations = iterations
        self.c = c
        self.rollout_depth = rollout_depth

    def move(
        self,
        board: np.ndarray,
        player: np.int8,
        state: Optional[object] = None
    ) -> Tuple[PlayerAction, Optional[object]]:
        # Root node: previous mover is opponent
        root = MCTSNode(
            board,
            parent=None,
            player=PLAYER2 if player == PLAYER1 else PLAYER1
        )

        for _ in range(self.iterations):
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # pick the child with highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return PlayerAction(best_move), None

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Traverse the tree using UCT until a leaf, expand if possible.
        """
        while node.children:
            total = sum(child.visits for child in node.children.values())
            # order children by UCT score
            node = max(
                node.children.values(),
                key=lambda n: n.uct_score(total, self.c)
            )
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Add one new child by exploring an untried move, in center-first order.
        """
        # prioritize center columns for expansion
        moves = sorted(node.valid_moves(), key=lambda c: abs(c - BOARD_COLS//2))
        for move in moves:
            if move not in node.children:
                new_board = node.board.copy()
                next_player = PLAYER2 if node.player == PLAYER1 else PLAYER1
                apply_player_action(new_board, PlayerAction(move), next_player)
                child = MCTSNode(new_board, parent=node, player=next_player)
                node.children[move] = child
                return child
        return node

    def _simulate(self, node: MCTSNode) -> float:
        """
        Perform a hybrid playout: heuristic moves up to rollout_depth, then static eval.
        Returns 1.0 if node.player eventual win, 0.0 if loss, 0.5 draw.
        """
        b = node.board.copy()
        current = PLAYER2 if node.player == PLAYER1 else PLAYER1
        for depth in range(self.rollout_depth):
            state = check_end_state(b, current)
            if state == GameState.IS_WIN:
                return 1.0 if current == node.player else 0.0
            if state == GameState.IS_DRAW:
                return 0.5
            # heuristic move: win, block, center, random
            # recompute valid moves for current board state
            valid = [
                c for c in range(BOARD_COLS)
                if check_move_status(b, np.int8(c)) == MoveStatus.IS_VALID
            ]
            move = self._heuristic_move(b, current, valid)
            apply_player_action(b, PlayerAction(move), current)
            current = PLAYER2 if current == PLAYER1 else PLAYER1
        # after rollout_depth, use static evaluation
        static = score_position(b, node.player)
        if static > 0:
            return 1.0
        if static < 0:
            return 0.0
        return 0.5
        static = score_position(b, node.player)
        if static > 0:
            return 1.0
        if static < 0:
            return 0.0
        return 0.5

    def _heuristic_move(self, board: np.ndarray, current: np.int8, valid: List[int]) -> int:
        """
        Heuristic move: win, block, center, then random.
        """
        # winning move
        for c in valid:
            temp = board.copy()
            apply_player_action(temp, PlayerAction(c), current)
            if check_end_state(temp, current) == GameState.IS_WIN:
                return c
        # block opponent
        opponent = PLAYER2 if current == PLAYER1 else PLAYER1
        for c in valid:
            temp = board.copy()
            apply_player_action(temp, PlayerAction(c), opponent)
            if check_end_state(temp, opponent) == GameState.IS_WIN:
                return c
        # center
        center = BOARD_COLS // 2
        if center in valid:
            return center
        # random
        return random.choice(valid)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Propagate simulation result up to the root.
        """
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent

# Expose agent
AgentMCTS: GenMove = MCTSAgent().move
