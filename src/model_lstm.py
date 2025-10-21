"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

import torch
from torch import nn


class _SequentialLSTMLayer(nn.Module):
    """Thin wrapper mimicking a single Keras ``LSTM`` layer."""

    def __init__(
        self,
        input_size: int,
        units: int,
        dropout_rate: float,
        return_sequences: bool,
    ) -> None:
        super().__init__()

        self.return_sequences = return_sequences
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=units,
            batch_first=True,
            dropout=dropout_rate,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


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

        layers: list[nn.Module] = []
        input_size = self.feature_size
        for layer_index in range(lstm_layers):
            return_sequences = layer_index < lstm_layers - 1
            layers.append(
                _SequentialLSTMLayer(
                    input_size=input_size,
                    units=units,
                    dropout_rate=dropout_rate,
                    return_sequences=return_sequences,
                )
            )
            input_size = units

        self.lstm_stack = nn.Sequential(*layers)
        self.classifier = nn.Linear(units, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass that mirrors the provided Keras design."""

        batch_size = inputs.size(0)
        sequences = inputs.view(batch_size, self.sequence_length, self.feature_size)

        final_output = self.lstm_stack(sequences)
        if final_output.dim() == 3:
            final_output = final_output[:, -1, :]

        logits = self.classifier(final_output)
        return logits
