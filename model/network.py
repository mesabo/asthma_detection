import torch
import torch.nn as nn

class ConvLSTMNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ConvLSTMNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((input_dim - 2) // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, time_steps, 1)
        x = x.permute(0, 2, 1)                  # (batch, 1, time_steps)
        x = self.conv1(x)                       # (batch, 32, time_steps-2)
        x = self.pool1(x)                       # (batch, 32, pooled_steps)
        x = x.permute(0, 2, 1)                  # (batch, pooled_steps, 32)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)
