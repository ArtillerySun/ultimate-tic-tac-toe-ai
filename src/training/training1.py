import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import State, load_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class UltimateTTTDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)  # Reshape y to (n_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

def train_model(model, train_loader, criterion, optimizer, num_epochs=50, device="cpu"):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs.to(device)
            targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

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
            
data = load_data()
# data = random.sample(data, 80000)
LEN = len(data)
print(LEN)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

y = (np.array(list(map(lambda x : x[1], data))) + 1) / 2.0
x_features = np.array([extract_features(data[i][0]) for i in range(LEN)])

x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=42)


print("start to train")

dataset = UltimateTTTDataset(x_train, y_train, device)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = UTTTNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, num_epochs=200, device=device)

torch.save(model.state_dict(), 'model/uttt_model.pth')

model.eval()

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

with torch.no_grad():
    y_pred = model(x_test_tensor)

criterion = torch.nn.MSELoss()
mse = criterion(y_pred, y_test_tensor).item()
rmse = np.sqrt(mse)

# Print results
print(f'Mean Squared Error (MSE) on test set: {mse:.4f}')
print(f'Root Mean Squared Error (RMSE) on test set: {rmse:.4f}')
