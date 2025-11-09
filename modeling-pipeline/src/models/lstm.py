import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden_size=256, num_layers=2, dropout=0.1, bidirectional=False, head_hidden=256, head_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        layers, h = [], out_dim
        for _ in range(head_layers-1):
            layers += [nn.Linear(h, head_hidden), nn.ReLU(), nn.Dropout(dropout)]
            h = head_hidden
        layers.append(nn.Linear(h, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x, month_idx=None, static=None):
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        return self.head(h).squeeze(-1)
