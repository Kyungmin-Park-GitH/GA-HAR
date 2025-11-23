"""Utility functions for loading and preparing multi-sensor HAR infrared datasets (3 sensors)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    """Container storing preprocessed samples and labels for a dataset split (3 sensors)."""

    samples_sensor1: np.ndarray
    samples_sensor2: np.ndarray
    samples_sensor3: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetInfo:
    """Stores all the information required to train on a dataset (3 sensors)."""

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
            indices.append(int(stripped) - 1)
    return indices


def _load_single_csv(path: str) -> Tuple[np.ndarray, int]:
    """Loads an individual CSV file and returns (sample, label)."""

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

    sample = values.reshape(frame_count, spatial_size, spatial_size)
    sample = _apply_gaussian_smoothing(sample)

    label_value = frame.iloc[0, values_per_frame]
    try:
        label = int(label_value)
    except ValueError as exc:
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


def load_single_sensor_data(
    dataset_path: str, test_index_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads a single sensor dataset and returns train/test samples and labels."""

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

    train_samples_array = np.stack(train_samples, axis=0).astype(np.float32)
    train_labels_array = np.array(train_labels, dtype=np.int64)
    test_samples_array = np.stack(test_samples, axis=0).astype(np.float32)
    test_labels_array = np.array(test_labels, dtype=np.int64)

    return train_samples_array, train_labels_array, test_samples_array, test_labels_array


def load_dataset_3sensors(
    coventry_sensor1_path: str,
    coventry_sensor2_path: str,
    coventry_sensor3_path: str,
    infra_sensor2_path: str,
    infra_sensor3_path: str,
    infra_sensor4_path: str,
    coventry_test_indices: str,
    infra_test_indices: str,
) -> Tuple[DatasetInfo, DatasetInfo]:
    """Loads datasets for 3-sensor training.

    Training combines:
    - Sensor A: coventry_sensor1 + infra_sensor3
    - Sensor B: coventry_sensor2 + infra_sensor4
    - Sensor C: coventry_sensor3 + infra_sensor2

    Test evaluates separately on:
    - coventry (sensor1, sensor2, sensor3)
    - infra (sensor3, sensor4, sensor2)

    Returns:
        Tuple of (coventry_dataset, infra_dataset) with 3-sensor structure.
    """

    # Load coventry sensors
    cov_s1_train, cov_s1_train_labels, cov_s1_test, cov_s1_test_labels = load_single_sensor_data(
        coventry_sensor1_path, coventry_test_indices
    )
    cov_s2_train, cov_s2_train_labels, cov_s2_test, cov_s2_test_labels = load_single_sensor_data(
        coventry_sensor2_path, coventry_test_indices
    )
    cov_s3_train, cov_s3_train_labels, cov_s3_test, cov_s3_test_labels = load_single_sensor_data(
        coventry_sensor3_path, coventry_test_indices
    )

    # Load infra sensors
    infra_s2_train, infra_s2_train_labels, infra_s2_test, infra_s2_test_labels = load_single_sensor_data(
        infra_sensor2_path, infra_test_indices
    )
    infra_s3_train, infra_s3_train_labels, infra_s3_test, infra_s3_test_labels = load_single_sensor_data(
        infra_sensor3_path, infra_test_indices
    )
    infra_s4_train, infra_s4_train_labels, infra_s4_test, infra_s4_test_labels = load_single_sensor_data(
        infra_sensor4_path, infra_test_indices
    )

    # Verify labels match for same dataset
    assert np.array_equal(cov_s1_train_labels, cov_s2_train_labels), "Coventry train labels mismatch (s1 vs s2)"
    assert np.array_equal(cov_s1_train_labels, cov_s3_train_labels), "Coventry train labels mismatch (s1 vs s3)"
    assert np.array_equal(cov_s1_test_labels, cov_s2_test_labels), "Coventry test labels mismatch (s1 vs s2)"
    assert np.array_equal(cov_s1_test_labels, cov_s3_test_labels), "Coventry test labels mismatch (s1 vs s3)"

    assert np.array_equal(infra_s2_train_labels, infra_s3_train_labels), "Infra train labels mismatch (s2 vs s3)"
    assert np.array_equal(infra_s2_train_labels, infra_s4_train_labels), "Infra train labels mismatch (s2 vs s4)"
    assert np.array_equal(infra_s2_test_labels, infra_s3_test_labels), "Infra test labels mismatch (s2 vs s3)"
    assert np.array_equal(infra_s2_test_labels, infra_s4_test_labels), "Infra test labels mismatch (s2 vs s4)"

    # Normalize each sensor independently using training data statistics
    def normalize(train_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = float(train_data.mean())
        std = float(train_data.std())
        if std < 1e-6:
            std = 1.0
        return (train_data - mean) / std, (test_data - mean) / std

    cov_s1_train, cov_s1_test = normalize(cov_s1_train, cov_s1_test)
    cov_s2_train, cov_s2_test = normalize(cov_s2_train, cov_s2_test)
    cov_s3_train, cov_s3_test = normalize(cov_s3_train, cov_s3_test)
    infra_s2_train, infra_s2_test = normalize(infra_s2_train, infra_s2_test)
    infra_s3_train, infra_s3_test = normalize(infra_s3_train, infra_s3_test)
    infra_s4_train, infra_s4_test = normalize(infra_s4_train, infra_s4_test)

    # Create combined training data
    # Sensor A: coventry_sensor1 combined with infra_sensor3
    # Sensor B: coventry_sensor2 combined with infra_sensor4
    # Sensor C: coventry_sensor3 combined with infra_sensor2
    combined_train_sensor1 = np.concatenate([cov_s1_train, infra_s3_train], axis=0)
    combined_train_sensor2 = np.concatenate([cov_s2_train, infra_s4_train], axis=0)
    combined_train_sensor3 = np.concatenate([cov_s3_train, infra_s2_train], axis=0)
    combined_train_labels = np.concatenate([cov_s1_train_labels, infra_s3_train_labels], axis=0)

    # Create coventry test dataset (3 sensors)
    coventry_dataset = DatasetInfo(
        name="coventry",
        train=DatasetSplit(
            samples_sensor1=combined_train_sensor1,
            samples_sensor2=combined_train_sensor2,
            samples_sensor3=combined_train_sensor3,
            labels=combined_train_labels,
        ),
        test=DatasetSplit(
            samples_sensor1=cov_s1_test,
            samples_sensor2=cov_s2_test,
            samples_sensor3=cov_s3_test,
            labels=cov_s1_test_labels,
        ),
        num_classes=11,
    )

    # Create infra test dataset (3 sensors)
    # Test uses: sensor3, sensor4, sensor2 (matching the training pairing order)
    infra_dataset = DatasetInfo(
        name="infra",
        train=DatasetSplit(
            samples_sensor1=combined_train_sensor1,
            samples_sensor2=combined_train_sensor2,
            samples_sensor3=combined_train_sensor3,
            labels=combined_train_labels,
        ),
        test=DatasetSplit(
            samples_sensor1=infra_s3_test,
            samples_sensor2=infra_s4_test,
            samples_sensor3=infra_s2_test,
            labels=infra_s3_test_labels,
        ),
        num_classes=11,
    )

    return coventry_dataset, infra_dataset
