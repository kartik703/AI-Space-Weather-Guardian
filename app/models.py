"""
models.py
----------
Neural network architectures for Space Weather Guardian.
Includes:
- GRUModel: Gated Recurrent Unit for sequence modeling
- TransformerModel: Transformer Encoder for advanced temporal learning
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU-based sequence model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through GRU.
        x: (batch, seq_len, input_size)
        """
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]   # take last timestep
        return self.fc(out)   # (batch, output_size)


class TransformerModel(nn.Module):
    """
    Transformer-based sequence model for time series forecasting.
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        Forward pass through Transformer.
        x: (batch, seq_len, input_size)
        """
        x = self.embedding(x)          # (batch, seq_len, d_model)
        out = self.transformer(x)      # (batch, seq_len, d_model)
        out = out[:, -1, :]            # take last timestep
        return self.fc(out)            # (batch, output_size)


# Explicit exports
__all__ = ["GRUModel", "TransformerModel"]
