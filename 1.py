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
 