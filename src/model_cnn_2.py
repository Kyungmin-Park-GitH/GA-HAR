"""Neural network definition for the infrared HAR task with 2 sensor inputs."""
from __future__ import annotations

import torch
from torch import nn


class HARConvNet2Sensor(nn.Module):
    """Convolutional network for 2-sensor 8x8 infrared inputs with late fusion."""

    def __init__(
        self,
        num_classes: int,
        conv_layers: int,
        filters: int,
        kernel_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        # Sensor 1 branch
        layers1 = []
        in_channels = 40
        for _ in range(conv_layers):
            layers1.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            layers1.append(nn.ReLU())
            in_channels = filters
        self.conv1 = nn.Sequential(*layers1)

        # Sensor 2 branch
        layers2 = []
        in_channels = 40
        for _ in range(conv_layers):
            layers2.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            layers2.append(nn.ReLU())
            in_channels = filters
        self.conv2 = nn.Sequential(*layers2)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        # Classifier takes concatenated features from both sensors
        self.classifier = nn.Linear(filters * 8 * 8 * 2, num_classes)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass with 2 sensor inputs.

        Args:
            input1: Tensor with shape ``(batch_size, 40, 8, 8)`` for sensor 1.
            input2: Tensor with shape ``(batch_size, 40, 8, 8)`` for sensor 2.
        """

        features1 = self.conv1(input1)
        features2 = self.conv2(input2)

        flat1 = torch.flatten(features1, start_dim=1)
        flat2 = torch.flatten(features2, start_dim=1)

        combined = torch.cat([flat1, flat2], dim=1)
        logits = self.classifier(self.dropout(combined))
        return logits
