from game_utils import *
from agents.random_agent import Agent as RandomAgent
from agents.minimax_agent import Agent as MinimaxAgent
from agents.mcts_agent import AgentMCTS

AGENTS = {
    '1': ('Human vs Human', None),
    '2': ('Human vs Random AI', RandomAgent),
    '3': ('Human vs Minimax AI', MinimaxAgent),
    '4': ('Human vs MCTS AI', AgentMCTS),
}


def main():
    # Prompt user to select a game mode
    print("Select mode:")
    for key, (desc, _) in AGENTS.items():
        print(f"  {key}. {desc}")
    choice = input(f"Enter 1-{len(AGENTS)}: ").strip()
    while choice not in AGENTS:
        choice = input("Invalid choice; enter 1-4: ").strip()

    desc, AgentCls = AGENTS[choice]
    print(f"→ {desc}")
    ai_agent = AgentCls  # selected AI or None for human

    board = initialize_game_state()  # empty board
    current = PLAYER1                # PLAYER1 always starts
    state_ai = None                  # placeholder for AI state

    # Game loop until win or draw
    while True:
        print(pretty_print_board(board))

        if ai_agent and current == PLAYER2:
            # AI's turn: call agent
            move, state_ai = ai_agent(board, current, state_ai)
            print(f"AI chooses column {int(move)}")
        else:
            # Human's turn: solicit input
            raw = input(f"Player {int(current)}, choose column (0–6): ").strip()
            try:
                move = PlayerAction(int(raw))
            except ValueError:
                print("Please enter an integer 0-6.")
                continue

        # Validate move before applying
        status = check_move_status(board, move)
        if status is not MoveStatus.IS_VALID:
            print(status.value)
            continue

        apply_player_action(board, move, current)
        result = check_end_state(board, current)
        if result == GameState.IS_WIN:
            print(pretty_print_board(board))
            print(f"Player {int(current)} wins!")
            break
        if result == GameState.IS_DRAW:
            print(pretty_print_board(board))
            print("It's a draw!")
            break

        # Alternate turn
        current = PLAYER2 if current == PLAYER1 else PLAYER1


if __name__ == "__main__":
    main()
