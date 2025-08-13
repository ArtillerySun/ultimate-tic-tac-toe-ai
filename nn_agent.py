import numpy as np
import time

import torch
import torch.nn as nn


from utils import State, Action
    
class UTTTNet(nn.Module):
    def __init__(self):
        super(UTTTNet, self).__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class StudentAgent:

    def __init__(self):
        """Instantiates your agent.
        """
        # with open("model/decision_tree_model.pkl", "rb") as f:
        # with open("model/sgd_model_params.pkl", "rb") as f:
        # with open("model/svm_model.pkl", "rb") as f:
            # self.loaded_model = pickle.load(f)
            # self.loaded_model, self.scaler = pickle.load(f)
        self.model = UTTTNet(input_size=30)

        self.model.load_state_dict(torch.load('model/uttt_model_new_new_new.pth', map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        action = self._minimax_alpha_beta(state.clone(), float('-inf'), float('inf'))
        return action

    def _minimax_alpha_beta(
        self,
        state: State,
        alpha: float,
        beta: float,
    ) -> Action:

        time_limit = 2.85
        start_time = time.time()
        max_depth = 1

        def evaluation(state: State):

            if state.is_terminal():
                return state.terminal_utility()

            def extract_features(state: State):
                def count_adjacent_pairs(board, player):
                    pairs = 0
                    
                    for i in range(3):
                        if board[i][0] == board[i][2] == player and board[i][1] == 0:
                            pairs += 1
                        if board[i][0] == board[i][1] == player and board[i][2] == 0:
                            pairs += 1
                        if board[i][1] == board[i][2] == player and board[i][0] == 0:
                            pairs += 1
                        
                        if board[0][i] == board[2][i] == player and board[1][i] == 0:
                            pairs += 1
                        if board[0][i] == board[1][i] == player and board[2][i] == 0:
                            pairs += 1
                        if board[1][i] == board[2][i] == player and board[0][i] == 0:
                            pairs += 1
                    
                    # Diagonal pairs
                    if board[0, 0] == board[1, 1] == player and board[2, 2] == 0:
                        pairs += 1
                    if board[0, 0] == board[2, 2] == player and board[1, 1] == 0:
                        pairs += 1
                    if board[1, 1] == board[2, 2] == player and board[0, 0] == 0:
                        pairs += 1

                    if board[0, 2] == board[1, 1] == player and board[2, 0] == 0:
                        pairs += 1
                    if board[0, 2] == board[2, 0] == player and board[1, 1] == 0:
                        pairs += 1
                    if board[1, 1] == board[2, 0] == player and board[0, 2] == 0:
                        pairs += 1

                    return pairs

                x = state.board
                local_status = state.local_board_status
                prev_action = np.array((-1, -1) if state.prev_local_action is None else state.prev_local_action)
                current_player = state.fill_num

                cnt1 = np.zeros((9))
                cnt2 = np.zeros((9))
                for i in range(3):
                    for j in range(3):
                        k = i * 3 + j

                        if local_status[i, j] != 0:
                            if local_status[i, j] == 1:
                                cnt1[k] = 10
                                cnt2[k] = 0
                            else:
                                cnt1[k] = 0
                                cnt2[k] = 10
                        else:
                            cnt1[k] = count_adjacent_pairs(x[i, j], 1)
                            cnt2[k] = count_adjacent_pairs(x[i, j], 2)
                            if x[i, j, 1, 1] != 0:
                                if x[i, j, 1, 1] == 1:
                                    cnt1[k] += 1
                                else:
                                    cnt2[k] += 1


                center_area_1 = np.sum(x[1, 1] == 1)
                center_area_2 = np.sum(x[1, 1] == 2)

                open_local_boards = np.sum(local_status == 0)

                k = prev_action[0] * 3 + prev_action[1]
                prev_action_influence1 = (1.5 if k == 4 else 1) * (cnt1[k] - cnt2[k])
                if k < 0:
                    if current_player == 1:
                        prev_action_influence1 = 5
                    else:
                        prev_action_influence1 = -5


                prev_action_influence2 = 0
                if k < 0:
                    if current_player == 1:
                        prev_action_influence2 = 5
                    else:
                        prev_action_influence2 = -5
                else:
                    prev_action_influence2 = cnt1[k] - cnt2[k]
                    if current_player == 1:
                        prev_action_influence2 += 5
                    else:
                        prev_action_influence2 -= 5
                    if k == 4:
                        prev_action_influence2 *= 1.5

                num_of_actions = len(state.get_all_valid_actions())
                num_to_win = 0
                oppo_local_pairs = 0
                for action in state.get_all_valid_actions():
                    new_state = state.change_state(action)
                    if new_state.is_terminal():
                        num_to_win += 1
                    
                    nxt_local = new_state.prev_local_action
                    if not nxt_local == None:
                        oppo_local_pairs += count_adjacent_pairs(x[nxt_local[0], nxt_local[1]], new_state.fill_num)
                    else:
                        for i in range(3):
                            for j in range(3):
                                oppo_local_pairs += count_adjacent_pairs(x[i, j], new_state.fill_num)

                num_to_win *= 1 if current_player == 1 else -1
                if num_of_actions > 0:
                    oppo_local_pairs /= num_of_actions
                oppo_local_pairs *= 1 if current_player == 1 else -1

                turn_number = np.sum(x != 0)

                local_pairs1 = count_adjacent_pairs(local_status, 1)
                local_pairs2 = count_adjacent_pairs(local_status, 2)

                x_features = np.concatenate([
                    # x.flatten(),  # Flatten the global board state
                    # local_status.flatten(),  # Flatten the local board status
                    [current_player],
                    cnt1,
                    cnt2,
                    [center_area_1],
                    [center_area_2],
                    [open_local_boards],
                    [prev_action_influence1],
                    [prev_action_influence2],
                    [num_of_actions],
                    [num_to_win],
                    [oppo_local_pairs],
                    [turn_number],
                    [local_pairs1],
                    [local_pairs2]
                ])
                
                return x_features

            feature = extract_features(state)
            x_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():  # No gradient computation for inference
                prediction = self.model(x_tensor)
            return prediction
            # prediction = self.loaded_model.predict([feature])
            # prediction = self.loaded_model.predict(self.scaler.transform([feature]))
            # return prediction[0]

        def max_value(
            state: State,
            depth: int,
            alpha: float,
            beta: float,
        ):
            if time.time() - start_time > time_limit:
                raise TimeoutError
                
            if state.is_terminal() or depth == max_depth:
                return (evaluation(state), None)

            v = float('-inf')
            target_action = None
            actions = state.get_all_valid_actions()
            for action in actions:
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
            if time.time() - start_time > time_limit:
                raise TimeoutError
                
            if state.is_terminal() or depth == max_depth:
                return (evaluation(state), None)
            
            v = float('inf')
            target_action = None
            actions = state.get_all_valid_actions()
            for action in actions:
                nxt_value, _ = max_value(state.change_state(action), depth + 1, alpha, beta)
                if nxt_value < v:
                    v = nxt_value
                    target_action = action
                beta = min(beta, v)
                if v <= alpha:
                    return (v, target_action)
                
            return (v, target_action)

        best_action = None
        while True:
            try:
                if state.fill_num == 1:
                    _, action = max_value(state, 0, alpha, beta)
                    best_action = action
                else:
                    _, action = min_value(state, 0, alpha, beta)
                    best_action = action

                max_depth += 1

            except TimeoutError:
                break

        return best_action
    