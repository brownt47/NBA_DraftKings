
"""PyTorch LSTM for predicting NBA DraftKings points."""

import torch
from torch import nn


class DraftKingsLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.network = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, sequences, lengths):
        # Sequences are right-aligned and left-padded.
        # Extract real games before packing.
        batch, steps, features = sequences.shape
        aligned = torch.zeros_like(sequences)

        for i in range(batch):
            count = int(lengths[i].item())
            if count:
                aligned[i, :count] = sequences[i, -count:]

        safe_lengths = lengths.clamp(min=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            aligned,
            safe_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        output = self.network(hidden[-1]).squeeze(-1)
        return output
