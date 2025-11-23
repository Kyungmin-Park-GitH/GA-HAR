"""Hybrid CNN-LSTM network definition for the infrared HAR task with 2 sensor inputs."""
from __future__ import annotations

import torch
from torch import nn


class HARHybridNet2Sensor(nn.Module):
    """CNN-LSTM hybrid for 2-sensor 8x8 infrared inputs with late fusion."""

    def __init__(
        self,
        num_classes: int,
        conv_layers: int,
        filters: int,
        kernel_size: int,
        lstm_layers: int,
        units: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        if conv_layers < 1:
            raise ValueError("Hybrid model requires at least one convolutional layer")
        if lstm_layers < 1:
            raise ValueError("Hybrid model requires at least one LSTM layer")

        # Sensor 1 CNN branch
        conv_blocks1 = []
        in_channels = 1
        for _ in range(conv_layers):
            conv_blocks1.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            conv_blocks1.append(nn.ReLU())
            in_channels = filters
        self.conv1 = nn.Sequential(*conv_blocks1)

        # Sensor 2 CNN branch
        conv_blocks2 = []
        in_channels = 1
        for _ in range(conv_layers):
            conv_blocks2.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            conv_blocks2.append(nn.ReLU())
            in_channels = filters
        self.conv2 = nn.Sequential(*conv_blocks2)

        self.frame_dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        feature_size = filters * 8 * 8

        # Sensor 1 LSTM
        self.lstm1 = nn.LSTM(
            input_size=feature_size,
            hidden_size=units,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # Sensor 2 LSTM
        self.lstm2 = nn.LSTM(
            input_size=feature_size,
            hidden_size=units,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.temporal_dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        # Classifier takes concatenated features from both sensors
        self.classifier = nn.Linear(units * 2, num_classes)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass with 2 sensor inputs.

        Args:
            input1: Tensor with shape ``(batch_size, 40, 8, 8)`` for sensor 1.
            input2: Tensor with shape ``(batch_size, 40, 8, 8)`` for sensor 2.
        """

        batch_size, sequence_length, height, width = input1.shape

        # Process sensor 1
        frames1 = input1.view(batch_size * sequence_length, 1, height, width)
        conv_features1 = self.conv1(frames1)
        conv_features1 = self.frame_dropout(conv_features1)
        feature_vectors1 = conv_features1.view(batch_size, sequence_length, -1)
        _, (hidden1, _) = self.lstm1(feature_vectors1)
        final_state1 = hidden1[-1]

        # Process sensor 2
        frames2 = input2.view(batch_size * sequence_length, 1, height, width)
        conv_features2 = self.conv2(frames2)
        conv_features2 = self.frame_dropout(conv_features2)
        feature_vectors2 = conv_features2.view(batch_size, sequence_length, -1)
        _, (hidden2, _) = self.lstm2(feature_vectors2)
        final_state2 = hidden2[-1]

        # Combine and classify
        combined = torch.cat([final_state1, final_state2], dim=1)
        activated = torch.tanh(combined)
        logits = self.classifier(self.temporal_dropout(activated))
        return logits
