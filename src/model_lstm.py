"""LSTM-based network definition for the infrared HAR task."""
from __future__ import annotations

import torch
from torch import nn


class _SequentialLSTMLayer(nn.Module):
    """Single LSTM layer with recurrent dropout support."""

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
        self.units = units
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=units)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs the temporal loop while applying recurrent dropout."""

        if inputs.dim() != 3:
            raise ValueError(
                "Expected inputs shaped as (batch, time, features); "
                f"received {tuple(inputs.shape)}"
            )

        batch_size, time_steps, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        hidden_state = torch.zeros(batch_size, self.units, device=device, dtype=dtype)
        cell_state = torch.zeros(batch_size, self.units, device=device, dtype=dtype)
        outputs: list[torch.Tensor] = []
        dropout_mask: torch.Tensor | None = None

        for step in range(time_steps):
            timestep_input = inputs[:, step, :]

            if self.training and self.dropout_rate > 0.0:
                if dropout_mask is None or dropout_mask.size(0) != batch_size:
                    dropout_mask = torch.bernoulli(
                        torch.full(
                            (batch_size, self.units),
                            1.0 - self.dropout_rate,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    dropout_mask.div_(max(1.0 - self.dropout_rate, torch.finfo(dtype).eps))
                hidden_for_cell = hidden_state * dropout_mask
            else:
                hidden_for_cell = hidden_state

            hidden_state, cell_state = self.cell(
                timestep_input,
                (hidden_for_cell, cell_state),
            )
            outputs.append(hidden_state.unsqueeze(1))

        sequence_outputs = torch.cat(outputs, dim=1)
        if self.return_sequences:
            return sequence_outputs
        return sequence_outputs[:, -1, :]


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
