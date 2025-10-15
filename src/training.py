"""Convenience re-exports for the active training utilities.

The LSTM workflow is active by default. To reuse the CNN version import from
:mod:`src.training_cnn` or uncomment the lines below.
"""
from __future__ import annotations

# from .training_cnn import (  # Enable for CNN-based experiments.
#     EvaluationMetrics,
#     TrainingConfig,
#     evaluate_with_kfold,
# )
from .training_lstm import EvaluationMetrics, TrainingConfig, evaluate_with_kfold

__all__ = ["EvaluationMetrics", "TrainingConfig", "evaluate_with_kfold"]
