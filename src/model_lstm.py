"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

import torch
from torch import nn


class _LSTMBlock(nn.Module):
    """Single LSTM layer with input dropout and built-in unrolling."""

    def __init__(
        self,
        input_size: int,
        units: int,
        dropout_rate: float,
        return_sequences: bool,
    ) -> None:
        super().__init__()

        self.return_sequences = return_sequences
        self.input_dropout: nn.Module
        if dropout_rate > 0.0:
            self.input_dropout = nn.Dropout(p=dropout_rate)
        else:
            self.input_dropout = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=units,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs the high-level LSTM layer and returns the desired output."""

        if inputs.dim() != 3:
            raise ValueError(
                "Expected inputs shaped as (batch, time, features); "
                f"received {tuple(inputs.shape)}"
            )

        dropped_inputs = self.input_dropout(inputs)
        outputs, _ = self.lstm(dropped_inputs)
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
                _LSTMBlock(
                    input_size=input_size,
                    units=units,
                    dropout_rate=dropout_rate,
                    return_sequences=return_sequences,
                )
            )
            input_size = units

        self.lstm_stack = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(units, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass that mirrors the provided Keras design."""

        batch_size = inputs.size(0)
        sequences = inputs.view(batch_size, self.sequence_length, self.feature_size)

        final_output = self.lstm_stack(sequences)
        if final_output.dim() == 3:
            final_output = final_output[:, -1, :]

        probabilities = self.classifier(final_output)
        return probabilities
