"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

import torch
from torch import nn


class _LSTMBlock(nn.Module):
    """Replicates a single Keras ``LSTM`` layer with input dropout."""

    def __init__(self, input_size: int, units: int, dropout_rate: float) -> None:
        super().__init__()

        self.input_dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=units,
            batch_first=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dropped = self.input_dropout(inputs)
        outputs, _ = self.lstm(dropped)
        return outputs


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

        self.lstm_blocks = nn.ModuleList()
        for layer_index in range(lstm_layers):
            input_size = self.feature_size if layer_index == 0 else units
            self.lstm_blocks.append(
                _LSTMBlock(
                    input_size=input_size,
                    units=units,
                    dropout_rate=dropout_rate,
                )
            )

        self.classifier = nn.Linear(units, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass that mirrors the provided Keras design."""

        batch_size = inputs.size(0)
        sequences = inputs.view(batch_size, self.sequence_length, self.feature_size)

        for block in self.lstm_blocks:
            sequences = block(sequences)

        final_output = sequences[:, -1, :]
        logits = self.classifier(final_output)
        return logits
