import torch, math
import torch.nn as nn

class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=48):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MonthEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.emb = nn.Embedding(12, d_model)

    def forward(self, x, month_idx):
        # month_idx: [B, T] with -1 for padded steps -> mask them as Jan
        month_idx = month_idx.clamp(min=0)
        return x + self.emb(month_idx)

class TransformerRegressor(nn.Module):
    def __init__(self, in_dim, d_model=256, nhead=8, n_layers=4, dim_ff=768, dropout=0.1,
                 posenc="sine", head_hidden=256, head_layers=2, static_dim=0, use_film=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.posenc_type = posenc
        if posenc == "sine":
            self.posenc = SinePositionalEncoding(d_model)
        elif posenc == "month_embed":
            self.posenc = MonthEmbedding(d_model)
        else:
            self.posenc = nn.Parameter(torch.zeros(1, 48, d_model))  # learned
        layers = []
        h = d_model
        for _ in range(head_layers-1):
            layers += [nn.Linear(h, head_hidden), nn.ReLU(), nn.Dropout(dropout)]
            h = head_hidden
        layers.append(nn.Linear(h, 1))
        self.head = nn.Sequential(*layers)

        self.static_dim = int(static_dim)
        if self.static_dim > 0:
            self.static_proj = nn.Linear(self.static_dim, d_model)
            self.static_token_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.static_proj = None
            self.static_token_bias = None

        self.use_film = use_film and self.static_dim > 0
        if self.use_film:
            self.film = nn.Linear(self.static_dim, in_dim * 2)
        else:
            self.film = None

    def forward(self, x, month_idx=None, static=None):
        # x: [B, T, F]
        if self.film is not None and static is not None and static.numel() > 0:
            gamma_beta = self.film(static)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        h_seq = self.proj(x)
        if isinstance(self.posenc, MonthEmbedding):
            h_seq = self.posenc(h_seq, month_idx)
        elif isinstance(self.posenc, SinePositionalEncoding):
            h_seq = self.posenc(h_seq)
        else:
            h_seq = h_seq + self.posenc[:, :h_seq.size(1)]

        if self.static_proj is not None and static is not None and static.numel() > 0:
            s = self.static_proj(static).unsqueeze(1)
            s = s + self.static_token_bias
            h = torch.cat([s, h_seq], dim=1)
        else:
            h = h_seq

        h = self.encoder(h)               # [B,T,d]
        h = h.mean(dim=1)                 # simple pooling
        y = self.head(h).squeeze(-1)
        return y
