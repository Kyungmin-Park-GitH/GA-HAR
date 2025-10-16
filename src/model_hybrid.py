"""Hybrid CNN-LSTM network definition for the infrared HAR task."""
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


class HARHybridNet(nn.Module):
    """Applies frame-wise convolutions followed by temporal LSTM aggregation."""

    def __init__(
        self,
        num_classes: int,
        conv_layers: int,
        filters: int,
        kernel_size: int,
        lstm_layers: int,
        units: int,
        dropout_rate: float,
        activation: str,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'.")
        if conv_layers < 1:
            raise ValueError("Hybrid model requires at least one convolutional layer")
        if lstm_layers < 1:
            raise ValueError("Hybrid model requires at least one LSTM layer")

        activation_factory = _ACTIVATIONS[activation]

        conv_blocks = []
        in_channels = 1  # process each frame independently
        for _ in range(conv_layers):
            conv_blocks.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            conv_blocks.append(activation_factory())
            in_channels = filters

        self.conv = nn.Sequential(*conv_blocks)
        self.frame_dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        feature_size = filters * 8 * 8
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=units,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.activation = activation_factory()
        self.temporal_dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(units, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass.

        Args:
            inputs: Tensor with shape ``(batch_size, 40, 8, 8)``.
        """

        batch_size, sequence_length, height, width = inputs.shape
        frames = inputs.view(batch_size * sequence_length, 1, height, width)
        conv_features = self.conv(frames)
        conv_features = self.frame_dropout(conv_features)
        feature_vectors = conv_features.view(batch_size, sequence_length, -1)

        _, (hidden, _) = self.lstm(feature_vectors)
        final_state = hidden[-1]
        activated = self.activation(final_state)
        logits = self.classifier(self.temporal_dropout(activated))
        return logits
