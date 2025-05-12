#!/usr/bin/env python3

from game_utils import (
    initialize_game_state,
    pretty_print_board,
    check_move_status,
    apply_player_action,
    check_end_state,
    PlayerAction,
    PLAYER1,
    PLAYER2,
    GameState,
    MoveStatus,
)
from agents.random_agent import Agent as RandomAgent
from agents.minimax_agent import Agent as MinimaxAgent
from agents.rl_agent import Agent as RLAgent
from agents.mcts_agent import AgentMCTS
# from agents.alphazero_agent import AgentAZ

# Map mode inputs to agent callables
AGENTS = {
    '1': ('Human vs Human', None),
    '2': ('Human vs Random AI', RandomAgent),
    '3': ('Human vs Minimax AI', MinimaxAgent),
    '4': ('Human vs RL AI', RLAgent),
    '5': ('Human vs MCTS AI', AgentMCTS),
    # '6': ('Human vs AlphaZero AI', AgentAZ(lambda b,p: [1/BOARD_COLS]*BOARD_COLS, lambda b,p: 0.0)),
}


def main():
    print("Select mode:")
    for key, (desc, _) in AGENTS.items():
        print(f"  {key}. {desc}")
    choice = input("Enter 1-6: ").strip()
    while choice not in AGENTS:
        choice = input("Invalid choice; enter 1-6: ").strip()

    desc, AgentCls = AGENTS[choice]
    print(f"→ {desc}")
    ai_agent = AgentCls if AgentCls is not None else None

    board = initialize_game_state()
    current = PLAYER1
    state_ai = None

    while True:
        print(pretty_print_board(board))
        if ai_agent and current == PLAYER2:
            move, state_ai = ai_agent(board, current, state_ai)
            print(f"AI chooses column {int(move)}")
        else:
            raw = input(f"Player {int(current)}, choose column (0–6): ").strip()
            try:
                move = PlayerAction(int(raw))
            except ValueError:
                print("Please enter an integer 0-6.")
                continue

        status = check_move_status(board, move)
        if status is not MoveStatus.IS_VALID:
            print(status.value)
            continue

        apply_player_action(board, move, current)
        state = check_end_state(board, current)
        if state == GameState.IS_WIN:
            print(pretty_print_board(board))
            print(f"Player {int(current)} wins!")
            break
        if state == GameState.IS_DRAW:
            print(pretty_print_board(board))
            print("It's a draw!")
            break

        current = PLAYER2 if current == PLAYER1 else PLAYER1

if __name__ == "__main__":
    from game_utils import BOARD_COLS
    main()