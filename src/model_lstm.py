"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

import torch
from torch import nn


class HARLSTMNet(nn.Module):
    """Stacked LSTM classifier operating on 8x8 infrared sequences."""

    def __init__(
        self,
        num_classes: int,
        lstm_layers: int,
        units: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        if lstm_layers < 1:
            raise ValueError("LSTM requires at least one layer")

        self.sequence_length = 40
        self.feature_size = 8 * 8
        self.units = units

        # ``batch_first`` allows receiving inputs shaped ``(batch, time, features)``.
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=units,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(units, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass."""

        batch_size = inputs.size(0)
        flattened = inputs.view(batch_size, self.sequence_length, self.feature_size)
        _, (hidden, _) = self.lstm(flattened)
        final_state = hidden[-1]
        activated = torch.tanh(final_state)
        logits = self.classifier(self.dropout(activated))
        return logits
