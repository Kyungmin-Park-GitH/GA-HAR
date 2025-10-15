"""Neural network definition for the infrared HAR task."""
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


class HARConvNet(nn.Module):
    """Simple convolutional network tailored to the 8x8 infrared inputs."""

    def __init__(
        self,
        num_classes: int,
        conv_layers: int,
        filters: int,
        kernel_size: int,
        dropout_rate: float,
        activation: str,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'.")

        activation_factory = _ACTIVATIONS[activation]

        layers = []
        in_channels = 40  # Each sample contains 40 frames stacked as channels.
        for _ in range(conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            layers.append(activation_factory())
            in_channels = filters

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(filters * 8 * 8, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass.

        Args:
            inputs: Tensor with shape ``(batch_size, 40, 8, 8)``.
        """

        features = self.conv(inputs)
        flattened = torch.flatten(features, start_dim=1)
        logits = self.classifier(self.dropout(flattened))
        return logits
