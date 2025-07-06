import torch
from torch import nn, sigmoid
class Predictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size=1):
        super().__init__()

        self.memory_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.3,
            num_layers=2,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=hidden_size*2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_size),
        )

    def forward(self, x):

        out, _ = self.memory_layer(x)

        output = out[:, -1, :]

        output = self.fc(output)

        return sigmoid(output)