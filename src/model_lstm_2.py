"""LSTM-based network definition for the infrared HAR task with 2 sensor inputs."""
from __future__ import annotations

import torch
from torch import nn


class _LSTMBlock(nn.Module):
    """Single LSTM layer with Keras-style input dropout."""

    def __init__(
        self,
        input_size: int,
        units: int,
        dropout_rate: float,
        return_sequences: bool,
    ) -> None:
        super().__init__()

        self.return_sequences = return_sequences
        self.dropout_rate = float(dropout_rate)
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0)")

        self.input_size = input_size

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

        if self.training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            if keep_prob <= 0.0:
                raise ValueError("dropout_rate of 1.0 disables the entire layer")

            mask = inputs.new_empty(inputs.size(0), 1, self.input_size)
            mask.bernoulli_(keep_prob).div_(keep_prob)
            dropped_inputs = inputs * mask
        else:
            dropped_inputs = inputs

        outputs, _ = self.lstm(dropped_inputs)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


class HARLSTMNet2Sensor(nn.Module):
    """Stacked LSTM classifier for 2-sensor 8x8 infrared sequences with late fusion."""

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

        # Sensor 1 LSTM stack
        layers1: list[nn.Module] = []
        input_size = self.feature_size
        for layer_index in range(lstm_layers):
            return_sequences = layer_index < lstm_layers - 1
            layers1.append(
                _LSTMBlock(
                    input_size=input_size,
                    units=units,
                    dropout_rate=dropout_rate,
                    return_sequences=return_sequences,
                )
            )
            input_size = units
        self.lstm_stack1 = nn.Sequential(*layers1)

        # Sensor 2 LSTM stack
        layers2: list[nn.Module] = []
        input_size = self.feature_size
        for layer_index in range(lstm_layers):
            return_sequences = layer_index < lstm_layers - 1
            layers2.append(
                _LSTMBlock(
                    input_size=input_size,
                    units=units,
                    dropout_rate=dropout_rate,
                    return_sequences=return_sequences,
                )
            )
            input_size = units
        self.lstm_stack2 = nn.Sequential(*layers2)

        # Classifier takes concatenated features from both sensors
        self.classifier = nn.Sequential(
            nn.Linear(units * 2, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass with 2 sensor inputs."""

        batch_size = input1.size(0)

        # Process sensor 1
        seq1 = input1.view(batch_size, self.sequence_length, self.feature_size)
        out1 = self.lstm_stack1(seq1)
        if out1.dim() == 3:
            out1 = out1[:, -1, :]

        # Process sensor 2
        seq2 = input2.view(batch_size, self.sequence_length, self.feature_size)
        out2 = self.lstm_stack2(seq2)
        if out2.dim() == 3:
            out2 = out2[:, -1, :]

        # Concatenate and classify
        combined = torch.cat([out1, out2], dim=1)
        probabilities = self.classifier(combined)
        return probabilities
