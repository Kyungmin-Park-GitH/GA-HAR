"""Training utilities tailored to the LSTM HAR model."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .model import HARLSTMNet


def _reshape_for_lstm(inputs: np.ndarray) -> np.ndarray:
    """Reshapes samples to ``(batch, time, features)`` for the LSTM."""

    if inputs.ndim == 4:
        batch_size, time_steps, height, width = inputs.shape
        return inputs.reshape(batch_size, time_steps, height * width)
    if inputs.ndim == 3:
        return inputs
    raise ValueError(
        "Expected inputs with 3 or 4 dimensions for LSTM training, "
        f"received shape {inputs.shape}."
    )


@dataclass
class TrainingConfig:
    """Configuration parameters shared across training runs."""

    learning_rate: float
    batch_size: int
    lstm_layers: int
    units: int
    dropout_rate: float
    max_epochs: int = 1000
    patience: int = 5


@dataclass
class EvaluationMetrics:
    """Container for classification metrics computed on the test set."""

    accuracy: float
    recall: float
    precision: float
    f1: float
    confusion_matrix: np.ndarray

    @staticmethod
    def average(metrics: Sequence["EvaluationMetrics"]) -> "EvaluationMetrics":
        if not metrics:
            raise ValueError("Cannot average an empty sequence of metrics")

        accuracy = float(np.mean([metric.accuracy for metric in metrics]))
        recall = float(np.mean([metric.recall for metric in metrics]))
        precision = float(np.mean([metric.precision for metric in metrics]))
        f1 = float(np.mean([metric.f1 for metric in metrics]))
        confusion = np.sum([metric.confusion_matrix for metric in metrics], axis=0)
        return EvaluationMetrics(
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            f1=f1,
            confusion_matrix=confusion,
        )


def _create_dataloader(
    inputs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    sequences = _reshape_for_lstm(inputs)
    tensor_x = torch.from_numpy(np.ascontiguousarray(sequences, dtype=np.float32))
    tensor_y = torch.from_numpy(labels)
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy


def _compute_classification_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> EvaluationMetrics:
    model.eval()
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy().tolist())
            references.extend(labels.cpu().numpy().tolist())

    references_array = np.array(references, dtype=np.int64)
    predictions_array = np.array(predictions, dtype=np.int64)

    labels = list(range(num_classes))
    conf_matrix = confusion_matrix(references_array, predictions_array, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        references_array,
        predictions_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    accuracy = float((predictions_array == references_array).mean())

    return EvaluationMetrics(
        accuracy=accuracy,
        recall=float(recall),
        precision=float(precision),
        f1=float(f1),
        confusion_matrix=conf_matrix,
    )


def train_with_validation(
    config: TrainingConfig,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    val_inputs: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> HARLSTMNet:
    model = HARLSTMNet(
        num_classes=num_classes,
        lstm_layers=config.lstm_layers,
        units=config.units,
        dropout_rate=config.dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = _create_dataloader(train_inputs, train_labels, config.batch_size, shuffle=True)
    val_loader = _create_dataloader(val_inputs, val_labels, config.batch_size, shuffle=False)

    best_model = copy.deepcopy(model)
    best_accuracy = 0.0
    epochs_without_improvement = 0

    for _ in range(config.max_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, val_accuracy = _evaluate(model, val_loader, device)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    return best_model


FoldProgressCallback = Callable[[str, int, EvaluationMetrics | None], None]


def evaluate_with_kfold(
    config: TrainingConfig,
    inputs: np.ndarray,
    labels: np.ndarray,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    folds: int = 5,
    progress_callback: FoldProgressCallback | None = None,
) -> Tuple[EvaluationMetrics, ...]:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    test_loader = _create_dataloader(test_inputs, test_labels, config.batch_size, shuffle=False)

    fold_metrics: list[EvaluationMetrics] = []
    for fold_index, (train_indices, val_indices) in enumerate(kfold.split(inputs)):
        if progress_callback is not None:
            progress_callback("start", fold_index, None)

        train_model = train_with_validation(
            config=config,
            train_inputs=inputs[train_indices],
            train_labels=labels[train_indices],
            val_inputs=inputs[val_indices],
            val_labels=labels[val_indices],
            num_classes=num_classes,
            device=device,
        )

        metrics = _compute_classification_metrics(train_model, test_loader, device, num_classes)
        if progress_callback is not None:
            progress_callback("end", fold_index, metrics)

        fold_metrics.append(metrics)

    return tuple(fold_metrics)
