"""Model selection helpers for HAR experiments.

The LSTM implementation is exported by default. The original CNN network is still
available in :mod:`src.model_cnn` if needed for comparison. Uncomment the import
below to expose it directly from this module.
"""
from __future__ import annotations

# from .model_cnn import HARConvNet  # Enable when optimising the CNN baseline.
from .model_lstm import HARLSTMNet

__all__ = ["HARLSTMNet"]
