# agents/rl_agent.py

import numpy as np
import pickle
from typing import Optional, Tuple, Dict
from game_utils import GenMove, PlayerAction, NO_PLAYER, BOARD_COLS, BOARD_ROWS, apply_player_action

class QLearningAgent:
    """
    A simple Q-learning agent for Connect Four using a tabular Q-table.
    The Q-table maps serialized board states to Q-values for each column.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        q_table_file: Optional[str] = None
    ):
        # Learning rate, discount factor, exploration probability
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: Dict[Tuple[int, ...], np.ndarray] = {}
        self.last_state: Optional[Tuple[int, ...]] = None
        self.last_action: Optional[int] = None
        if q_table_file:
            try:
                with open(q_table_file, 'rb') as f:
                    self.q = pickle.load(f)
            except FileNotFoundError:
                pass

    def serialize(self, board: np.ndarray) -> Tuple[int, ...]:
        """Convert board array to a hashable state representation."""
        return tuple(board.flatten().tolist())

    def get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Retrieve or initialize Q-values array for a given state."""
        if state not in self.q:
            self.q[state] = np.zeros(BOARD_COLS, dtype=float)
        return self.q[state]

    def choose_action(self, board: np.ndarray) -> int:
        """Select an action using epsilon-greedy over valid columns."""
        state = self.serialize(board)
        q_vals = self.get_q_values(state)
        # Determine valid columns by checking top row
        valid_cols = [c for c in range(BOARD_COLS) if board[BOARD_ROWS - 1, c] == NO_PLAYER]
        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(valid_cols))
        else:
            # Mask invalid columns to -inf so they are never chosen
            mask = np.full(BOARD_COLS, -np.inf, dtype=float)
            mask[valid_cols] = q_vals[valid_cols]
            action = int(np.argmax(mask))
        # Store for learning update
        self.last_state = state
        self.last_action = action
        return action

    def learn(self, reward: float, new_board: np.ndarray, done: bool):
        """
        Update Q-table based on observed reward and next state.
        Should be called externally after applying an action and observing reward.
        """
        if self.last_state is None or self.last_action is None:
            return
        new_state = self.serialize(new_board)
        q_vals = self.get_q_values(self.last_state)
        next_q = 0.0
        if not done:
            next_q = np.max(self.get_q_values(new_state))
        # Q-learning update rule
        q_vals[self.last_action] += self.alpha * (
            reward + self.gamma * next_q - q_vals[self.last_action]
        )
        if done:
            self.last_state = None
            self.last_action = None

# Single global agent instance if no state passed
_default_agent = QLearningAgent()

def rl_move(
    board: np.ndarray,
    player: np.int8,
    state: Optional[QLearningAgent] = None
) -> Tuple[PlayerAction, Optional[QLearningAgent]]:
    """
    GenMove interface for QLearningAgent: returns (action, updated agent instance).
    Learning must be invoked externally by calling agent.learn().
    """
    agent = state if state is not None else _default_agent
    action = agent.choose_action(board)
    return PlayerAction(action), agent

Agent: GenMove = rl_move
