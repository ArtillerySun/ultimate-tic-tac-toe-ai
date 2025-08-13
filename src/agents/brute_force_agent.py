import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import State, Action


class StudentAgent:

    def __init__(self):
        return

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

        def evaluation(state: State):

            def check(a, b):
                return a >= 0 and b >= 0
            
            if state.is_terminal():
                value = state.terminal_utility()
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