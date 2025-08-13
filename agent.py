# Run the following cell to import utilities

import pickle
import numpy as np
import time

import torch
import torch.nn as nn

from sklearn.linear_model import SGDRegressor

from utils import State, Action, load_data
    
class UTTTNet(nn.Module):
    def __init__(self, input_size=26):
        super(UTTTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)          # Second hidden layer
        self.fc3 = nn.Linear(32, 1)           # Output layer (single value)
        self.relu = nn.ReLU()                 # Activation function
        self.dropout = nn.Dropout(0.2)        # Dropout for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here; we’ll use loss-appropriate activation
        return x

# data = load_data()
# assert len(data) == 80000
# for state, value in data[1000:1003]:
#     print(state)
#     print(f"Value = {value}\n\n")

class StudentAgent:

    def __init__(self):
        """Instantiates your agent.
        """
        # with open("model/decision_tree_model.pkl", "rb") as f:
        # with open("model/sgd_model_params.pkl", "rb") as f:
        # with open("model/svm_model.pkl", "rb") as f:
            # self.loaded_model = pickle.load(f)
            # self.loaded_model, self.scaler = pickle.load(f)
        self.model = UTTTNet(input_size=26)

        self.model.load_state_dict(torch.load('uttt_model.pth'))
        self.model.eval()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        _, action = self._minimax_alpha_beta(state.clone(), 3, float('-inf'), float('inf'))
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
                                cnt1[k] = 5
                                cnt2[k] = 0
                            else:
                                cnt1[k] = 0
                                cnt2[k] = 5
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
                prev_action_influence = (1.5 if k == 4 else 1) * (cnt1[k] - cnt2[k])

                num_of_actions = len(state.get_all_valid_actions())
                num_to_win = 0
                for action in state.get_all_valid_actions():
                    new_state = state.change_state(action)
                    if new_state.is_terminal():
                        num_to_win += 1
                num_to_win *= 1 if current_player == 1 else -1

                turn_number = np.sum(x != 0)        

                x_features = np.concatenate([
                    # x.flatten(),  # Flatten the global board state
                    # local_status.flatten(),  # Flatten the local board status
                    [current_player],
                    cnt1,
                    cnt2,
                    [center_area_1],
                    [center_area_2],
                    [open_local_boards],
                    [prev_action_influence],
                    [num_of_actions],
                    [num_to_win],
                    [turn_number]
                ])
                
                return x_features
            
            feature = extract_features(state)
            x_tensor = torch.tensor(feature, dtype=torch.float32)
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