# Connect 4 AI Project

This is a prototype for Connect 4 for the class Programming Course Project. The goal of this project is to implement and compare different AI agents to play the Connect 4 game.

## Files and Folders

- `game_utils.py`: Helper functions for the game.
- `agents/`: Contains AI agents:
  - `random_agent.py`: Makes random moves.
  - `minimax_agent.py`: Uses the Minimax algorithm.
  - `mcts_agent.py`: Monte Carlo Tree Search agent.
- `tests/`: Unit tests for the project.

## How to Run


 Run the game interface (the easiest way):
   ```bash
   python main.py
   ```
   One can play between two persons or between a person and an AI agent.


## Testing
 Run tests using `pytest`:
   ```bash
   pytest tests/
   ```
## Requirements

To set up the environment, it is recommended to use a virtual environment. First, create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # on Linux
```

Then, install the required dependencies, including `pytest` for running tests:

```bash
pip install -r requirements.txt
```


