# Connect 4 AI Project

This is a simple Connect 4 project for an assignment. It includes different AI agents to play the game.

## Files and Folders

- `main.py`: Run the game.
- `game_utils.py`: Helper functions for the game.
- `agents/`: Contains AI agents:
  - `random_agent.py`: Makes random moves.
  - `minimax_agent.py`: Uses the Minimax algorithm.
  - `mcts_agent.py`: Monte Carlo Tree Search agent.
  - `rl_agent.py`: Reinforcement Learning agent.
  - `agent_xy/`: Custom agent implementation.
- `tests/`: Unit tests for the project.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd connect4
   ```

2. Run the game:
   ```bash
   python main.py
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## Requirements

- Python 3.12 or higher
- Install required packages:
  ```bash
  pip install pytest
  ```

## Notes

Feel free to modify the agents or add your own custom agent in the `agent_xy` folder.
