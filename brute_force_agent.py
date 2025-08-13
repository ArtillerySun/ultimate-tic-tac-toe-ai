# Run the following cell to import utilities

import numpy as np
import time

from utils import State, Action, load_data

# data = load_data()
# assert len(data) == 80000
# for state, value in data[:3]:
#     print(state)
#     print(f"Value = {value}\n\n")

class StudentAgent:

    def __init__(self):
        """Instantiates your agent.
        """

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        _, action = self._minimax_alpha_beta(state.clone(), 4, float('-inf'), float('inf'))
        return action

    def _minimax_alpha_beta(
        self,
        state: State,
        max_depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Action]:
        # print(state)

        def evaluation(state: State):

            def check(a, b):
                return a >= 0 and b >= 0
            
            if state.is_terminal():
                value = state.terminal_utility()
                # if state.fill_num == 2:
                #     value = -value
                return value
            
            local_status = state.local_board_status

            macro_value = 0
            micro_value = 0
            for i in range(3):
                for j in range(3):
                    if local_status[i][j] == state.fill_num:
                        macro_value += 5
                    if local_status[i][j] == 2:
                        macro_value -= 5               

                    if check(i - 1, j - 1):
                        if local_status[i - 1][j - 1] == local_status[i, j] == 1:
                            macro_value += 10
                        if local_status[i - 1][j - 1] == local_status[i, j] == 2:
                            macro_value -= 10

                    if check(i - 1, j):
                        if local_status[i - 1][j] == local_status[i, j] == 1:
                            macro_value += 10
                        if local_status[i - 1][j] == local_status[i, j] == 2:
                            macro_value -= 10
                        
                    if check(i, j - 1):
                        if local_status[i][j - 1] == local_status[i, j] == 1:
                            macro_value += 10
                        if local_status[i][j - 1] == local_status[i, j] == 2:
                            macro_value -= 10

                    small_board = state.board[i][j]
                    for k in range(3):
                        for l in range(3):
                            if small_board[k][l] == 1:
                                micro_value += 1
                            if small_board[k][l] == 2:
                                micro_value -= 1

                            if check(k - 1, l - 1):
                                if small_board[k - 1][l - 1] == small_board[k][l] == 1:
                                    micro_value += 3
                                if small_board[k - 1][l - 1] == small_board[k][l] == 2:
                                    micro_value -= 3
                            
                            if check(k - 1, l):
                                if small_board[k - 1][l] == small_board[k][l] == 1:
                                    micro_value += 3
                                if small_board[k - 1][l] == small_board[k][l] == 2:
                                    micro_value -= 3

                            if check(k, l - 1):
                                if small_board[k][l - 1] == small_board[k][l] == 1:
                                    micro_value += 3
                                if small_board[k][l - 1] == small_board[k][l] == 2:
                                    micro_value -= 3

            w1 = 1
            w2 = 1
            value = w1 * macro_value + w2 * micro_value
            # if state.fill_num == 2:
            #     value = -value
            value = 1 / (1 + np.exp(-value))
            return value

        def max_value(
            state: State,
            depth: int,
            alpha: float,
            beta: float,
        ):
                
            if state.is_terminal() or depth == max_depth:
                return (evaluation(state), None)

            v = float('-inf')
            target_action = None
            for action in state.get_all_valid_actions():
                nxt_value, _ = min_value(state.change_state(action), depth + 1, alpha, beta)
                if nxt_value > v:
                    v = nxt_value
                    target_action = action
                alpha = max(alpha, v)
                if v >= beta:
                    return (v, target_action)
                
            return (v, target_action)

        def min_value(
            state: State,
            depth: int,
            alpha: float,
            beta: float,
        ):
                
            if state.is_terminal() or depth == max_depth:
                return (evaluation(state), None)
            
            v = float('inf')
            target_action = None
            for action in state.get_all_valid_actions():
                nxt_value, _ = max_value(state.change_state(action), depth + 1, alpha, beta)
                if nxt_value < v:
                    v = nxt_value
                    target_action = action
                beta = min(beta, v)
                if v <= alpha:
                    return (v, target_action)
                
            return (v, target_action)

        
        if state.fill_num == 1:
            return max_value(state, 0, alpha, beta)
        else:
            return min_value(state, 0, alpha, beta)
        """ YOUR CODE END HERE """
    
state = State(
    board=np.array([
        [
            [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
        ],
    ]),
    fill_num=1,
    prev_action=(2, 2, 0, 1),
)
# start_time = time.time()
# student_agent = StudentAgent()
# constructor_time = time.time()
# action = student_agent.choose_action(state)
# end_time = time.time()
# assert state.is_valid_action(action)
# print(f"Constructor time: {constructor_time - start_time}")
# print(f"Action time: {end_time - constructor_time}")
# assert constructor_time - start_time < 1
# assert end_time - constructor_time < 3

state = State(
    board=np.array([
        [
            [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
    ]),
    fill_num=1,
    prev_action=(2, 2, 0, 0)
)
# start_time = time.time()
# student_agent = StudentAgent()
# constructor_time = time.time()
# action = student_agent.choose_action(state)
# end_time = time.time()
# assert state.is_valid_action(action)
# print(f"Constructor time: {constructor_time - start_time}")
# print(f"Action time: {end_time - constructor_time}")
# assert constructor_time - start_time < 1
# assert end_time - constructor_time < 3