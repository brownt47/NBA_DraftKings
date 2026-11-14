
"""Hybrid LSTM combining player history and pregame features."""

import torch
from torch import nn


class DraftKingsHybridLSTM(nn.Module):
    def __init__(
        self,
        sequence_features=11,
        context_features=10,
        hidden_size=64,
        num_layers=1,
        dropout=0.20,
    ):
        super().__init__()

        # Historical player-game sequence.
        self.lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Upcoming-game context.
        self.context_network = nn.Sequential(
            nn.Linear(context_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combine historical and pregame information.
        self.output_network = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequences, lengths, context):
        batch, steps, features = sequences.shape

        # Existing sequences are left-padded.
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

        history = hidden[-1]

        # Prevent artificial zero-history sequences
        # from generating learned historical signals.
        history = history * (
            lengths > 0
        ).unsqueeze(1).to(history.dtype)

        game_context = self.context_network(context)

        combined = torch.cat(
            [history, game_context],
            dim=1,
        )

        return self.output_network(
            combined
        ).squeeze(-1)
