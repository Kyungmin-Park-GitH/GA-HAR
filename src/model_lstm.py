"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn


_ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "swish": lambda: nn.SiLU(inplace=False),
}


class HARLSTMNet(nn.Module):
    """Stacked LSTM classifier operating on 8x8 infrared sequences."""

    def __init__(
        self,
        num_classes: int,
        lstm_layers: int,
        units: int,
        dropout_rate: float,
        activation: str,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'.")

        if lstm_layers < 1:
            raise ValueError("LSTM requires at least one layer")

        activation_factory = _ACTIVATIONS[activation]

        self.sequence_length = 40
        self.feature_size = 8 * 8
        self.units = units
        self.activation = activation_factory()

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
        activated = self.activation(final_state)
        logits = self.classifier(self.dropout(activated))
        return logits
