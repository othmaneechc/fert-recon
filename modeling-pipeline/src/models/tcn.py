import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, nin, nout, k=3, d=1, dropout=0.1):
        super().__init__()
        pad = (k-1) * d
        self.net = nn.Sequential(
            nn.Conv1d(nin, nout, k, padding=pad, dilation=d),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(nout, nout, k, padding=pad, dilation=d),
            nn.ReLU(), nn.Dropout(dropout),
            Chomp1d(2*pad)
        )
        self.down = nn.Conv1d(nin, nout, 1) if nin != nout else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        return torch.relu(out + self.down(x))

class TCNRegressor(nn.Module):
    def __init__(self, in_dim, channels=[64,128,256], kernel_size=3, dropout=0.1, head_hidden=256, head_layers=2):
        super().__init__()
        layers, d, nin = [], 1, in_dim
        for ch in channels:
            layers.append(TemporalBlock(nin, ch, k=kernel_size, d=d, dropout=dropout))
            nin = ch; d *= 2
        self.tcn = nn.Sequential(*layers)
        h = channels[-1]
        mlp = []
        for _ in range(head_layers-1):
            mlp += [nn.Linear(h, head_hidden), nn.ReLU(), nn.Dropout(dropout)]
            h = head_hidden
        mlp.append(nn.Linear(h, 1))
        self.head = nn.Sequential(*mlp)

    def forward(self, x, month_idx=None, static=None):
        # x: [B,T,F] -> [B,F,T]
        h = self.tcn(x.transpose(1,2)).mean(dim=2)
        return self.head(h).squeeze(-1)
