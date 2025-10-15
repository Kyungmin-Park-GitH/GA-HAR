"""Entry points for selecting the desired GA backend.

By default the LSTM-based optimiser is exposed via ``NSGA2``. To switch back to the
CNN implementation import ``NSGA2`` from :mod:`src.ga_cnn` instead or uncomment the
line below.
"""
from __future__ import annotations

# from .ga_cnn import NSGA2 as NSGA2_CNN  # Enable for CNN-based optimisation.
from .ga_lstm import NSGA2 as NSGA2_LSTM

NSGA2 = NSGA2_LSTM
__all__ = ["NSGA2"]
