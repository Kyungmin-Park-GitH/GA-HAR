"""
3-sensor Hybrid model training script with 50 iterations.
Tracks per-sample test frequency and correctness for both coventry and infra datasets.
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model_hybrid_3 import HARHybridNet3Sensor


# =============================================================================
# Data Loading Utilities
# =============================================================================

def _gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Creates a 2D Gaussian kernel."""
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
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


def _load_single_csv(path: str) -> Tuple[np.ndarray, int]:
    """Loads an individual CSV file and returns (sample, label)."""
    frame_count = 40
    spatial_size = 8
    values_per_frame = spatial_size * spatial_size

    frame = pd.read_csv(path, header=None)
    values = frame.iloc[:, :values_per_frame].to_numpy(dtype=np.float32)
    sample = values.reshape(frame_count, spatial_size, spatial_size)
    sample = _apply_gaussian_smoothing(sample)

    label = int(frame.iloc[0, values_per_frame])
    return sample, label


def load_all_data_from_folder(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all CSV files from a folder and return samples and labels."""
    csv_files = sorted(
        [
            os.path.join(dataset_path, file_name)
            for file_name in os.listdir(dataset_path)
            if file_name.lower().endswith(".csv")
        ]
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    samples: List[np.ndarray] = []
    labels: List[int] = []

    for csv_path in csv_files:
        sample, label = _load_single_csv(csv_path)
        samples.append(sample)
        labels.append(label)

    return np.stack(samples, axis=0).astype(np.float32), np.array(labels, dtype=np.int64)


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize data and return normalized data with mean and std."""
    mean = float(data.mean())
    std = float(data.std())
    if std < 1e-6:
        std = 1.0
    return (data - mean) / std, mean, std


def apply_normalization(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply pre-computed normalization to data."""
    return (data - mean) / std


# =============================================================================
# Model Training Utilities
# =============================================================================

def create_dataloader_3sensor(
    inputs1: np.ndarray, inputs2: np.ndarray, inputs3: np.ndarray,
    labels: np.ndarray, batch_size: int, shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for 3-sensor inputs."""
    tensor_x1 = torch.from_numpy(inputs1)
    tensor_x2 = torch.from_numpy(inputs2)
    tensor_x3 = torch.from_numpy(inputs3)
    tensor_y = torch.from_numpy(labels)
    dataset = TensorDataset(tensor_x1, tensor_x2, tensor_x3, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model and return accuracy."""
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for inputs1, inputs2, inputs3, labels in data_loader:
            inputs1 = inputs1.to(device=device, dtype=torch.float32)
            inputs2 = inputs2.to(device=device, dtype=torch.float32)
            inputs3 = inputs3.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)
            outputs = model(inputs1, inputs2, inputs3)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / max(total_samples, 1)


def get_predictions(
    model: nn.Module,
    inputs1: np.ndarray, inputs2: np.ndarray, inputs3: np.ndarray,
    device: torch.device, batch_size: int
) -> np.ndarray:
    """Get model predictions for all samples."""
    model.eval()
    predictions_list = []

    # Create dataloader with dummy labels
    dummy_labels = np.zeros(len(inputs1), dtype=np.int64)
    data_loader = create_dataloader_3sensor(inputs1, inputs2, inputs3, dummy_labels, batch_size, shuffle=False)

    with torch.no_grad():
        for inputs1_batch, inputs2_batch, inputs3_batch, _ in data_loader:
            inputs1_batch = inputs1_batch.to(device=device, dtype=torch.float32)
            inputs2_batch = inputs2_batch.to(device=device, dtype=torch.float32)
            inputs3_batch = inputs3_batch.to(device=device, dtype=torch.float32)
            outputs = model(inputs1_batch, inputs2_batch, inputs3_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions_list.extend(preds)

    return np.array(predictions_list, dtype=np.int64)


def train_model(
    train_inputs1: np.ndarray, train_inputs2: np.ndarray, train_inputs3: np.ndarray,
    train_labels: np.ndarray,
    val_inputs1: np.ndarray, val_inputs2: np.ndarray, val_inputs3: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    conv_layers: int = 1,
    filters: int = 64,
    kernel_size: int = 6,
    lstm_layers: int = 1,
    units: int = 128,
    dropout_rate: float = 0.05,
    learning_rate: float = 0.0025,
    batch_size: int = 64,
    max_epochs: int = 1000,
    patience: int = 5,
) -> HARHybridNet3Sensor:
    """Train a 3-sensor Hybrid model with early stopping."""
    model = HARHybridNet3Sensor(
        num_classes=num_classes,
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        lstm_layers=lstm_layers,
        units=units,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = create_dataloader_3sensor(
        train_inputs1, train_inputs2, train_inputs3, train_labels, batch_size, shuffle=True
    )
    val_loader = create_dataloader_3sensor(
        val_inputs1, val_inputs2, val_inputs3, val_labels, batch_size, shuffle=False
    )

    best_model = copy.deepcopy(model)
    best_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for inputs1, inputs2, inputs3, labels in train_loader:
            inputs1 = inputs1.to(device=device, dtype=torch.float32)
            inputs2 = inputs2.to(device=device, dtype=torch.float32)
            inputs3 = inputs3.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            optimizer.zero_grad()
            outputs = model(inputs1, inputs2, inputs3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_accuracy = evaluate_model(model, val_loader, device)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return best_model


# =============================================================================
# Main Script
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-sensor Hybrid model 50-iteration training")
    parser.add_argument(
        "--coventry-sensor1-path",
        default="./data/coventry_2018/40_linear_sensor1",
        help="Path to the Coventry sensor 1 dataset directory.",
    )
    parser.add_argument(
        "--coventry-sensor2-path",
        default="./data/coventry_2018/40_linear_sensor2",
        help="Path to the Coventry sensor 2 dataset directory.",
    )
    parser.add_argument(
        "--coventry-sensor3-path",
        default="./data/coventry_2018/40_linear_sensor3",
        help="Path to the Coventry sensor 3 dataset directory.",
    )
    parser.add_argument(
        "--infra-sensor2-path",
        default="./data/infra_adl2018/40_sensor2",
        help="Path to the INFRA sensor 2 dataset directory.",
    )
    parser.add_argument(
        "--infra-sensor3-path",
        default="./data/infra_adl2018/40_sensor3",
        help="Path to the INFRA sensor 3 dataset directory.",
    )
    parser.add_argument(
        "--infra-sensor4-path",
        default="./data/infra_adl2018/40_sensor4",
        help="Path to the INFRA sensor 4 dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to save result files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed model hyperparameters
    CONV_LAYERS = 1
    FILTERS = 64
    KERNEL_SIZE = 6
    LSTM_LAYERS = 1
    UNITS = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0025
    DROPOUT = 0.05
    NUM_CLASSES = 11
    NUM_ITERATIONS = 50

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Load all data from folders
    # =========================================================================
    print("Loading Coventry data...")
    cov_s1_samples, cov_s1_labels = load_all_data_from_folder(args.coventry_sensor1_path)
    cov_s2_samples, cov_s2_labels = load_all_data_from_folder(args.coventry_sensor2_path)
    cov_s3_samples, cov_s3_labels = load_all_data_from_folder(args.coventry_sensor3_path)

    print("Loading Infra data...")
    infra_s2_samples, infra_s2_labels = load_all_data_from_folder(args.infra_sensor2_path)
    infra_s3_samples, infra_s3_labels = load_all_data_from_folder(args.infra_sensor3_path)
    infra_s4_samples, infra_s4_labels = load_all_data_from_folder(args.infra_sensor4_path)

    # Verify labels match within each dataset
    assert np.array_equal(cov_s1_labels, cov_s2_labels), "Coventry labels mismatch (s1 vs s2)"
    assert np.array_equal(cov_s1_labels, cov_s3_labels), "Coventry labels mismatch (s1 vs s3)"
    assert np.array_equal(infra_s2_labels, infra_s3_labels), "Infra labels mismatch (s2 vs s3)"
    assert np.array_equal(infra_s2_labels, infra_s4_labels), "Infra labels mismatch (s2 vs s4)"

    cov_labels = cov_s1_labels
    infra_labels = infra_s3_labels

    n_cov = len(cov_labels)
    n_infra = len(infra_labels)

    print(f"Coventry samples: {n_cov}, Infra samples: {n_infra}")

    # =========================================================================
    # Initialize tracking arrays
    # =========================================================================
    # Track how many times each sample was used as test data
    cov_test_count = np.zeros(n_cov, dtype=np.int32)
    infra_test_count = np.zeros(n_infra, dtype=np.int32)

    # Track how many times each sample was correctly predicted when tested
    cov_correct_count = np.zeros(n_cov, dtype=np.int32)
    infra_correct_count = np.zeros(n_infra, dtype=np.int32)

    # =========================================================================
    # Run 50 iterations
    # =========================================================================
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")

        # ---------------------------------------------------------------------
        # Split Coventry data: 20% test, then 20% of remaining for validation
        # ---------------------------------------------------------------------
        cov_indices = np.arange(n_cov)
        np.random.shuffle(cov_indices)

        cov_test_size = int(n_cov * 0.2)
        cov_test_indices = cov_indices[:cov_test_size]
        cov_remain_indices = cov_indices[cov_test_size:]

        cov_val_size = int(len(cov_remain_indices) * 0.2)
        cov_val_indices = cov_remain_indices[:cov_val_size]
        cov_train_indices = cov_remain_indices[cov_val_size:]

        # ---------------------------------------------------------------------
        # Split Infra data: 20% test, then 20% of remaining for validation
        # ---------------------------------------------------------------------
        infra_indices = np.arange(n_infra)
        np.random.shuffle(infra_indices)

        infra_test_size = int(n_infra * 0.2)
        infra_test_indices = infra_indices[:infra_test_size]
        infra_remain_indices = infra_indices[infra_test_size:]

        infra_val_size = int(len(infra_remain_indices) * 0.2)
        infra_val_indices = infra_remain_indices[:infra_val_size]
        infra_train_indices = infra_remain_indices[infra_val_size:]

        # ---------------------------------------------------------------------
        # Update test counts
        # ---------------------------------------------------------------------
        cov_test_count[cov_test_indices] += 1
        infra_test_count[infra_test_indices] += 1

        # ---------------------------------------------------------------------
        # Prepare training data (combine coventry train + infra train)
        # Sensor mapping:
        #   Sensor1: cov_s1 + infra_s3
        #   Sensor2: cov_s2 + infra_s4
        #   Sensor3: cov_s3 + infra_s2
        # ---------------------------------------------------------------------
        train_sensor1 = np.concatenate([cov_s1_samples[cov_train_indices], infra_s3_samples[infra_train_indices]], axis=0)
        train_sensor2 = np.concatenate([cov_s2_samples[cov_train_indices], infra_s4_samples[infra_train_indices]], axis=0)
        train_sensor3 = np.concatenate([cov_s3_samples[cov_train_indices], infra_s2_samples[infra_train_indices]], axis=0)
        train_labels = np.concatenate([cov_labels[cov_train_indices], infra_labels[infra_train_indices]], axis=0)

        # Normalize using training data statistics
        train_sensor1, mean1, std1 = normalize_data(train_sensor1)
        train_sensor2, mean2, std2 = normalize_data(train_sensor2)
        train_sensor3, mean3, std3 = normalize_data(train_sensor3)

        # ---------------------------------------------------------------------
        # Prepare validation data (combine coventry val + infra val)
        # ---------------------------------------------------------------------
        val_sensor1 = np.concatenate([cov_s1_samples[cov_val_indices], infra_s3_samples[infra_val_indices]], axis=0)
        val_sensor2 = np.concatenate([cov_s2_samples[cov_val_indices], infra_s4_samples[infra_val_indices]], axis=0)
        val_sensor3 = np.concatenate([cov_s3_samples[cov_val_indices], infra_s2_samples[infra_val_indices]], axis=0)
        val_labels = np.concatenate([cov_labels[cov_val_indices], infra_labels[infra_val_indices]], axis=0)

        val_sensor1 = apply_normalization(val_sensor1, mean1, std1)
        val_sensor2 = apply_normalization(val_sensor2, mean2, std2)
        val_sensor3 = apply_normalization(val_sensor3, mean3, std3)

        # ---------------------------------------------------------------------
        # Prepare test data (separate for coventry and infra)
        # ---------------------------------------------------------------------
        # Coventry test: sensor1, sensor2, sensor3
        cov_test_sensor1 = apply_normalization(cov_s1_samples[cov_test_indices], mean1, std1)
        cov_test_sensor2 = apply_normalization(cov_s2_samples[cov_test_indices], mean2, std2)
        cov_test_sensor3 = apply_normalization(cov_s3_samples[cov_test_indices], mean3, std3)
        cov_test_labels = cov_labels[cov_test_indices]

        # Infra test: sensor3, sensor4, sensor2 (matching training pairing)
        infra_test_sensor1 = apply_normalization(infra_s3_samples[infra_test_indices], mean1, std1)
        infra_test_sensor2 = apply_normalization(infra_s4_samples[infra_test_indices], mean2, std2)
        infra_test_sensor3 = apply_normalization(infra_s2_samples[infra_test_indices], mean3, std3)
        infra_test_labels = infra_labels[infra_test_indices]

        # ---------------------------------------------------------------------
        # Train model
        # ---------------------------------------------------------------------
        print(f"Training... (train={len(train_labels)}, val={len(val_labels)})")
        model = train_model(
            train_sensor1, train_sensor2, train_sensor3, train_labels,
            val_sensor1, val_sensor2, val_sensor3, val_labels,
            num_classes=NUM_CLASSES,
            device=device,
            conv_layers=CONV_LAYERS,
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            lstm_layers=LSTM_LAYERS,
            units=UNITS,
            dropout_rate=DROPOUT,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
        )

        # ---------------------------------------------------------------------
        # Test on Coventry test set
        # ---------------------------------------------------------------------
        cov_predictions = get_predictions(
            model, cov_test_sensor1, cov_test_sensor2, cov_test_sensor3, device, BATCH_SIZE
        )
        cov_correct_mask = (cov_predictions == cov_test_labels)
        cov_correct_count[cov_test_indices[cov_correct_mask]] += 1

        cov_accuracy = cov_correct_mask.sum() / len(cov_correct_mask)
        print(f"Coventry Test Accuracy: {cov_accuracy:.4f} ({cov_correct_mask.sum()}/{len(cov_correct_mask)})")

        # ---------------------------------------------------------------------
        # Test on Infra test set
        # ---------------------------------------------------------------------
        infra_predictions = get_predictions(
            model, infra_test_sensor1, infra_test_sensor2, infra_test_sensor3, device, BATCH_SIZE
        )
        infra_correct_mask = (infra_predictions == infra_test_labels)
        infra_correct_count[infra_test_indices[infra_correct_mask]] += 1

        infra_accuracy = infra_correct_mask.sum() / len(infra_correct_mask)
        print(f"Infra Test Accuracy: {infra_accuracy:.4f} ({infra_correct_mask.sum()}/{len(infra_correct_mask)})")

    # =========================================================================
    # Save results
    # =========================================================================
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    # Coventry result file
    # Columns: index (1-based), label, test_count, correct_count
    cov_result = np.column_stack([
        np.arange(1, n_cov + 1),  # 1-based index
        cov_labels,
        cov_test_count,
        cov_correct_count,
    ])
    cov_result_path = os.path.join(args.output_dir, "coventry_result.csv")
    np.savetxt(cov_result_path, cov_result, delimiter=",", fmt="%d",
               header="index,label,test_count,correct_count", comments="")
    print(f"Saved: {cov_result_path}")

    # Infra result file
    infra_result = np.column_stack([
        np.arange(1, n_infra + 1),  # 1-based index
        infra_labels,
        infra_test_count,
        infra_correct_count,
    ])
    infra_result_path = os.path.join(args.output_dir, "infra_result.csv")
    np.savetxt(infra_result_path, infra_result, delimiter=",", fmt="%d",
               header="index,label,test_count,correct_count", comments="")
    print(f"Saved: {infra_result_path}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")

    cov_avg_test = cov_test_count.mean()
    cov_avg_correct_rate = np.where(cov_test_count > 0, cov_correct_count / cov_test_count, 0).mean()
    print(f"Coventry - Avg test count: {cov_avg_test:.2f}, Avg correct rate: {cov_avg_correct_rate:.4f}")

    infra_avg_test = infra_test_count.mean()
    infra_avg_correct_rate = np.where(infra_test_count > 0, infra_correct_count / infra_test_count, 0).mean()
    print(f"Infra - Avg test count: {infra_avg_test:.2f}, Avg correct rate: {infra_avg_correct_rate:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
