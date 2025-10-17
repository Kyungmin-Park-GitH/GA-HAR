"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

import torch
import torch.nn.functional as F
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
        self.dropout_rate = dropout_rate

        self.lstm_layers = nn.ModuleList()

        for layer_index in range(lstm_layers):
            input_size = self.feature_size if layer_index == 0 else units
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=units,
                    num_layers=1,
                    batch_first=True,
                )
            )

        self.classifier = nn.Linear(units, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass that mirrors the provided Keras design."""

        batch_size = inputs.size(0)
        sequences = inputs.view(batch_size, self.sequence_length, self.feature_size)

        for lstm in self.lstm_layers:
            if self.dropout_rate > 0.0:
                sequences = F.dropout(sequences, p=self.dropout_rate, training=self.training)
            sequences, (hidden, _) = lstm(sequences)

        final_output = sequences[:, -1, :]
        logits = self.classifier(final_output)
        return logits
