# =============================
# File: model.py
# =============================
import torch
import torch.nn as nn

class GoRankRNN(nn.Module):
    """
    Bidirectional LSTM encoder over variable-length move sequences.
    Input:  (B, T, F)
    Output: logits (B, 9)
    """
    def __init__(self, input_dim=36, hidden_dim=128, num_layers=2, num_classes=9, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, lengths):
        # x: [B, T, F], lengths: [B]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: [num_layers*2, B, H]
        # concat last layer's forward/backward states
        h_last_f = h_n[-2]
        h_last_b = h_n[-1]
        h = torch.cat([h_last_f, h_last_b], dim=1)  # [B, 2H]
        return self.head(h)





