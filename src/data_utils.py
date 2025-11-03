"""Utility functions for loading and preparing the HAR infrared datasets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    """Container storing preprocessed samples and labels for a dataset split."""

    samples: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetInfo:
    """Stores all the information required to train on a dataset."""

    name: str
    train: DatasetSplit
    test: DatasetSplit
    num_classes: int


def _read_test_indices(test_index_path: str) -> List[int]:
    """Reads the test indices file and returns zero-based indices."""

    with open(test_index_path, "r", encoding="utf-8") as file:
        indices: List[int] = []
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            # Indices in the specification are one-based, so convert to zero-based.
            indices.append(int(stripped) - 1)
    return indices


def _load_single_csv(path: str) -> Tuple[np.ndarray, int]:
    """Loads an individual CSV file and returns (sample, label).

    Each CSV contains 40 frames of 8x8 infrared data (64 values) and a label in
    the last column. The same label is repeated for all rows in the file.
    """

    frame_count = 40
    spatial_size = 8
    values_per_frame = spatial_size * spatial_size

    frame = pd.read_csv(path, header=None)
    if frame.shape[1] < values_per_frame + 1:
        raise ValueError(
            f"Expected at least {values_per_frame + 1} columns, found {frame.shape[1]} in {path}"
        )

    values = frame.iloc[:, :values_per_frame].to_numpy(dtype=np.float32)
    if values.shape[0] != frame_count:
        raise ValueError(
            f"Expected {frame_count} frames, found {values.shape[0]} in {path}"
        )

    # Reshape the 64 values per frame into an 8x8 grid. The final sample shape is
    # (frame_count, height, width).
    sample = values.reshape(frame_count, spatial_size, spatial_size)

    sample = _apply_gaussian_smoothing(sample)

    label_value = frame.iloc[0, values_per_frame]
    try:
        label = int(label_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Label value '{label_value}' in {path} is not an integer") from exc

    if not 0 <= label <= 10:
        raise ValueError(
            f"Label value {label} in {path} is outside the expected 0-10 range."
        )

    return sample, label


def _gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Creates a 2D Gaussian kernel."""

    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd to have a central element.")

    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0.0:
        raise ValueError("Gaussian kernel sum is zero, sigma may be too small.")
    kernel /= kernel_sum
    return kernel.astype(np.float32)


_GAUSSIAN_KERNEL = _gaussian_kernel()


def _apply_gaussian_smoothing(sample: np.ndarray) -> np.ndarray:
    """Applies Gaussian smoothing to each frame of the sample."""

    kernel = _GAUSSIAN_KERNEL
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    padded = np.pad(sample, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    smoothed = np.empty_like(sample, dtype=np.float32)

    for frame_idx in range(sample.shape[0]):
        frame = padded[frame_idx]
        for row in range(sample.shape[1]):
            for col in range(sample.shape[2]):
                window = frame[row : row + kernel_size, col : col + kernel_size]
                smoothed[frame_idx, row, col] = float(np.sum(window * kernel))

    return smoothed


def load_dataset(dataset_path: str, test_index_path: str, name: str) -> DatasetInfo:
    """Loads a dataset from disk and returns the processed splits.

    Args:
        dataset_path: Directory containing CSV files with HAR infrared data.
        test_index_path: Path to the file describing which samples belong to the
            held-out test set. Indices are one-based and refer to the
            lexicographical ordering of the CSV file names.
        name: Name used to identify the dataset in logs.
    """

    csv_files = sorted(
        [
            os.path.join(dataset_path, file_name)
            for file_name in os.listdir(dataset_path)
            if file_name.lower().endswith(".csv")
        ]
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    test_indices = set(_read_test_indices(test_index_path))

    train_samples: List[np.ndarray] = []
    train_labels: List[int] = []
    test_samples: List[np.ndarray] = []
    test_labels: List[int] = []

    for index, csv_path in enumerate(csv_files):
        sample, label = _load_single_csv(csv_path)
        if index in test_indices:
            test_samples.append(sample)
            test_labels.append(label)
        else:
            train_samples.append(sample)
            train_labels.append(label)

    if not train_samples:
        raise ValueError(
            "Training split is empty. Check that the test indices do not cover all files."
        )
    if not test_samples:
        raise ValueError(
            "Test split is empty. Ensure the test index file lists at least one sample."
        )

    train_samples_array = np.stack(train_samples, axis=0).astype(np.float32)
    train_labels_array = np.array(train_labels, dtype=np.int64)
    test_samples_array = np.stack(test_samples, axis=0).astype(np.float32)
    test_labels_array = np.array(test_labels, dtype=np.int64)

    mean = float(train_samples_array.mean())
    std = float(train_samples_array.std())
    if std < 1e-6:
        std = 1.0

    train_samples_array = (train_samples_array - mean) / std
    test_samples_array = (test_samples_array - mean) / std

    return DatasetInfo(
        name=name,
        train=DatasetSplit(samples=train_samples_array, labels=train_labels_array),
        test=DatasetSplit(samples=test_samples_array, labels=test_labels_array),
        num_classes=11,
    )
