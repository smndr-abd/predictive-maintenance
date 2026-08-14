"""
models.py
---------
Two architectures for Remaining Useful Life (RUL) regression:
  1. LSTMRegressor       - stacked LSTM + dense head
  2. TransformerRegressor - Transformer encoder + dense head

Both take input of shape (batch, seq_len, n_features) and output a single
scalar RUL prediction per sequence, shape (batch,).
"""

import math
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = out[:, -1, :]          # final timestep's hidden state
        rul = self.head(last_hidden).squeeze(-1)
        return rul


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.2, max_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)          # (batch, seq_len, d_model)
        pooled = h.mean(dim=1)       # mean pooling over time
        rul = self.head(pooled).squeeze(-1)
        return rul


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMRegressor(n_features, **kwargs)
    elif name == "transformer":
        return TransformerRegressor(n_features, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    # Quick shape sanity check
    batch, seq_len, n_features = 8, 30, 17
    dummy = torch.randn(batch, seq_len, n_features)

    lstm = build_model("lstm", n_features)
    out = lstm(dummy)
    print("LSTM output shape:", out.shape)

    transformer = build_model("transformer", n_features)
    out2 = transformer(dummy)
    print("Transformer output shape:", out2.shape)
