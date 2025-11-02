"""Multi-objective genetic algorithm (NSGA-II) for LSTM-based HAR model search."""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .data_utils import DatasetInfo
from .training import EvaluationMetrics, TrainingConfig, evaluate_with_kfold


HYPERPARAMETER_SPACE: Dict[str, Sequence] = {
    "learning_rate": [i / 10000 for i in range(1, 101)],
    "batch_size":    [8, 16, 32, 48, 64, 96, 128, 192, 256],
    "lstm_layers":   [1, 2, 3, 4, 5],
    "units":         [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "dropout_rate": [i / 100 for i in range(0, 51)],
}


def _compute_objectives(
    fold_metrics_by_dataset: Dict[str, Tuple[EvaluationMetrics, ...]]
) -> Tuple[float, float, float, float]:
    """Computes the four NSGA-II objectives from per-fold metrics."""

    if not fold_metrics_by_dataset:
        return 0.0, 0.0, 0.0, 0.0

    metrics_sequences = list(fold_metrics_by_dataset.values())
    fold_count = len(metrics_sequences[0]) if metrics_sequences else 0
    if fold_count == 0:
        return 0.0, 0.0, 0.0, 0.0

    for metrics in metrics_sequences:
        if len(metrics) != fold_count:
            raise ValueError("All datasets must provide the same number of folds")

    mean_accuracy_values: list[float] = []
    min_accuracy_values: list[float] = []
    mean_f1_values: list[float] = []
    min_f1_values: list[float] = []

    for fold_index in range(fold_count):
        accuracies = [metrics[fold_index].accuracy for metrics in metrics_sequences]
        f1_scores = [metrics[fold_index].f1 for metrics in metrics_sequences]

        mean_accuracy_values.append(float(sum(accuracies) / len(accuracies)))
        min_accuracy_values.append(float(min(accuracies)))
        mean_f1_values.append(float(sum(f1_scores) / len(f1_scores)))
        min_f1_values.append(float(min(f1_scores)))

    fold_count_float = float(fold_count)
    return (
        sum(mean_accuracy_values) / fold_count_float,
        sum(min_accuracy_values) / fold_count_float,
        sum(mean_f1_values) / fold_count_float,
        sum(min_f1_values) / fold_count_float,
    )


def _slugify(value: str) -> str:
    """Converts arbitrary dataset names to filesystem-friendly slugs."""

    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "dataset"

def _random_value(key: str) -> object:
    """Samples a random value for the provided hyperparameter."""

    return random.choice(HYPERPARAMETER_SPACE[key])


@dataclass(eq=True, frozen=True)
class Genome:
    """Representation of a candidate solution in the GA."""

    learning_rate: float
    batch_size: int
    lstm_layers: int
    units: int
    dropout_rate: float

    def crossover(self, other: "Genome") -> "Genome":
        """Performs uniform crossover between two parents with 50% mixing."""

        params = {}
        for field_name in self.__dataclass_fields__:
            if random.random() < 0.5:
                params[field_name] = getattr(self, field_name)
            else:
                params[field_name] = getattr(other, field_name)
        return Genome(**params)

    def mutate(self, mutation_rate: float = 0.1) -> "Genome":
        """Mutates each hyperparameter with ``mutation_rate`` probability."""

        params = {}
        for field_name in self.__dataclass_fields__:
            current_value = getattr(self, field_name)
            if random.random() < mutation_rate:
                candidates = [value for value in HYPERPARAMETER_SPACE[field_name] if value != current_value]
                params[field_name] = random.choice(candidates) if candidates else current_value
            else:
                params[field_name] = current_value
        return Genome(**params)

    @staticmethod
    def random() -> "Genome":
        """Creates a genome with randomly sampled hyperparameters."""

        params = {key: _random_value(key) for key in HYPERPARAMETER_SPACE}
        return Genome(**params)


@dataclass
class Individual:
    """Wraps a genome and stores evaluation results."""

    genome: Genome
    fitness: Tuple[float, float, float, float] | None = None
    crowding_distance: float = 0.0
    rank: int = math.inf
    metrics_by_dataset: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    fold_metrics_by_dataset: Dict[str, Tuple[EvaluationMetrics, ...]] = field(default_factory=dict)

    def clone(self) -> "Individual":
        return Individual(
            genome=self.genome,
            fitness=self.fitness,
            crowding_distance=self.crowding_distance,
            rank=self.rank,
            metrics_by_dataset=self.metrics_by_dataset.copy(),
            fold_metrics_by_dataset=self.fold_metrics_by_dataset.copy(),
        )


class EvaluationLogger:
    """Handles writing evaluation results to CSV files after each generation."""

    def __init__(self, output_dir: str, datasets: Sequence[DatasetInfo]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.datasets = list(datasets)
        self.dataset_slugs = {dataset.name: _slugify(dataset.name) for dataset in datasets}
        self.hyperparameter_fields = list(Genome.__dataclass_fields__.keys())

        self.detail_files = {
            dataset.name: self.output_dir / f"{self.dataset_slugs[dataset.name]}_details.csv"
            for dataset in datasets
        }
        self.confusion_files = {
            dataset.name: self.output_dir / f"{self.dataset_slugs[dataset.name]}_confusion.csv"
            for dataset in datasets
        }
        self.overall_file = self.output_dir / "overall_results.csv"

        for path in [*self.detail_files.values(), *self.confusion_files.values(), self.overall_file]:
            if path.exists():
                path.unlink()

    @staticmethod
    def _write_header(path: Path, header: Sequence[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    @staticmethod
    def _append_row(path: Path, row: Sequence[object]) -> None:
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def log_generation(self, generation: int, individuals: Sequence[Individual]) -> None:
        """Writes per-fold and aggregated metrics for the selected individuals."""

        if not individuals:
            return

        sorted_individuals = sorted(individuals, key=lambda ind: (ind.rank, -ind.crowding_distance))

        for dataset in self.datasets:
            detail_path = self.detail_files[dataset.name]
            if not detail_path.exists():
                header = [
                    "generation",
                    "individual",
                    "fold",
                    *self.hyperparameter_fields,
                    "accuracy",
                    "recall",
                    "precision",
                    "f1_score",
                ]
                self._write_header(detail_path, header)

            confusion_path = self.confusion_files[dataset.name]
            if not confusion_path.exists():
                confusion_header = [
                    "generation",
                    "individual",
                    "fold",
                    *[
                        f"cm_{i}_{j}"
                        for i in range(dataset.num_classes)
                        for j in range(dataset.num_classes)
                    ],
                ]
                self._write_header(confusion_path, confusion_header)

        if not self.overall_file.exists():
            header = ["generation", "individual", *self.hyperparameter_fields]
            for dataset in self.datasets:
                slug = self.dataset_slugs[dataset.name]
                header.extend(
                    [
                        f"{slug}_accuracy",
                        f"{slug}_recall",
                        f"{slug}_precision",
                        f"{slug}_f1_score",
                    ]
                )
            header.extend([
                "mean_accuracy_objective",
                "min_accuracy_objective",
                "mean_f1_objective",
                "min_f1_objective",
            ])
            self._write_header(self.overall_file, header)

        for rank_index, individual in enumerate(sorted_individuals, start=1):
            if not individual.metrics_by_dataset or not individual.fold_metrics_by_dataset:
                raise RuntimeError(
                    "Evaluation results missing for an individual despite being selected."
                )

            for dataset in self.datasets:
                fold_metrics = individual.fold_metrics_by_dataset.get(dataset.name)
                if not fold_metrics:
                    continue

                detail_row_prefix = [
                    generation,
                    rank_index,
                    None,  # placeholder for fold index
                    *[getattr(individual.genome, field) for field in self.hyperparameter_fields],
                ]

                confusion_row_prefix = [
                    generation,
                    rank_index,
                    None,
                ]

                for fold_index, metrics in enumerate(fold_metrics, start=1):
                    detail_row = detail_row_prefix.copy()
                    detail_row[2] = fold_index
                    detail_row.extend(
                        [
                            f"{metrics.accuracy:.6f}",
                            f"{metrics.recall:.6f}",
                            f"{metrics.precision:.6f}",
                            f"{metrics.f1:.6f}",
                        ]
                    )
                    self._append_row(self.detail_files[dataset.name], detail_row)

                    confusion_row = confusion_row_prefix.copy()
                    confusion_row[2] = fold_index
                    confusion_row.extend(metrics.confusion_matrix.astype(int).flatten().tolist())
                    self._append_row(self.confusion_files[dataset.name], confusion_row)

            overall_row: List[object] = [generation, rank_index]
            overall_row.extend(getattr(individual.genome, field) for field in self.hyperparameter_fields)
            for dataset in self.datasets:
                metrics = individual.metrics_by_dataset.get(dataset.name)
                if metrics is None:
                    continue
                overall_row.extend(
                    [
                        f"{metrics.accuracy:.6f}",
                        f"{metrics.recall:.6f}",
                        f"{metrics.precision:.6f}",
                        f"{metrics.f1:.6f}",
                    ]
                )
            objectives = _compute_objectives(individual.fold_metrics_by_dataset)
            overall_row.extend([f"{value:.6f}" for value in objectives])
            self._append_row(self.overall_file, overall_row)


class Evaluator:
    """Handles the evaluation of genomes on the available datasets."""

    def __init__(
        self,
        datasets: Sequence[DatasetInfo],
        device: torch.device,
    ) -> None:
        self.datasets = datasets
        self.device = device

    def evaluate(
        self,
        genome: Genome,
        generation: int,
        individual_index: int,
        total_individuals: int,
    ) -> Tuple[Dict[str, EvaluationMetrics], Dict[str, Tuple[EvaluationMetrics, ...]]]:
        """Evaluates ``genome`` and returns aggregated and per-fold metrics."""

        config = TrainingConfig(
            learning_rate=genome.learning_rate,
            batch_size=genome.batch_size,
            lstm_layers=genome.lstm_layers,
            units=genome.units,
            dropout_rate=genome.dropout_rate,
        )

        combined_train_inputs = np.concatenate(
            [dataset.train.samples for dataset in self.datasets], axis=0
        )
        combined_train_labels = np.concatenate(
            [dataset.train.labels for dataset in self.datasets], axis=0
        )
        test_sets = {
            dataset.name: (dataset.test.samples, dataset.test.labels)
            for dataset in self.datasets
        }

        def progress_callback(
            event: str,
            fold_index: int,
            dataset_name: str | None,
            metrics: EvaluationMetrics | None,
        ) -> None:
            prefix = (
                f"[Generation {generation:03d}] Individual {individual_index + 1:02d}/"
                f"{total_individuals:02d} Fold {fold_index + 1}/5"
            )
            if event == "start":
                print(f"{prefix}: training...", flush=True)
            elif dataset_name is not None and metrics is not None:
                print(
                    (
                        f"{prefix} test on '{dataset_name}': acc={metrics.accuracy:.4f}, "
                        f"recall={metrics.recall:.4f}, precision={metrics.precision:.4f}, "
                        f"f1={metrics.f1:.4f}"
                    ),
                    flush=True,
                )

        num_classes = max(dataset.num_classes for dataset in self.datasets)

        fold_metrics_by_dataset = evaluate_with_kfold(
            config=config,
            inputs=combined_train_inputs,
            labels=combined_train_labels,
            test_sets=test_sets,
            num_classes=num_classes,
            device=self.device,
            folds=5,
            progress_callback=progress_callback,
        )

        metrics_by_dataset: Dict[str, EvaluationMetrics] = {}
        for dataset in self.datasets:
            dataset_metrics = fold_metrics_by_dataset.get(dataset.name)
            if not dataset_metrics:
                continue
            metrics_by_dataset[dataset.name] = EvaluationMetrics.average(dataset_metrics)

        return metrics_by_dataset, fold_metrics_by_dataset


def _non_dominated_sort(population: Sequence[Individual]) -> List[List[Individual]]:
    """Performs non-dominated sorting and returns fronts."""

    fronts: List[List[Individual]] = []
    domination_counts: Dict[int, int] = {i: 0 for i in range(len(population))}
    dominated: Dict[int, List[int]] = {i: [] for i in range(len(population))}

    for i, individual in enumerate(population):
        for j, other in enumerate(population):
            if i == j:
                continue
            if dominates(individual, other):
                dominated[i].append(j)
            elif dominates(other, individual):
                domination_counts[i] += 1

    current_front = [i for i, count in domination_counts.items() if count == 0]
    front_index = 0
    while current_front:
        for idx in current_front:
            population[idx].rank = front_index
        fronts.append([population[idx] for idx in current_front])
        next_front: List[int] = []
        for idx in current_front:
            for dominated_index in dominated[idx]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    next_front.append(dominated_index)
        front_index += 1
        current_front = next_front
    return fronts


def dominates(individual_a: Individual, individual_b: Individual) -> bool:
    """Checks whether ``individual_a`` Pareto-dominates ``individual_b``."""

    if individual_a.fitness is None or individual_b.fitness is None:
        raise ValueError("Individuals must be evaluated before dominance comparison.")

    better_or_equal = all(a >= b for a, b in zip(individual_a.fitness, individual_b.fitness))
    strictly_better = any(a > b for a, b in zip(individual_a.fitness, individual_b.fitness))
    return better_or_equal and strictly_better


def _compute_crowding_distance(front: Sequence[Individual]) -> None:
    """Computes crowding distance for a front in-place."""

    if not front:
        return

    num_objectives = len(front[0].fitness or [])
    for individual in front:
        individual.crowding_distance = 0.0

    for objective in range(num_objectives):
        front_sorted = sorted(front, key=lambda ind: ind.fitness[objective] if ind.fitness else 0.0)
        front_sorted[0].crowding_distance = float("inf")
        front_sorted[-1].crowding_distance = float("inf")
        min_value = front_sorted[0].fitness[objective]
        max_value = front_sorted[-1].fitness[objective]
        if math.isclose(max_value, min_value):
            continue
        for i in range(1, len(front_sorted) - 1):
            prev_value = front_sorted[i - 1].fitness[objective]
            next_value = front_sorted[i + 1].fitness[objective]
            distance = (next_value - prev_value) / (max_value - min_value)
            front_sorted[i].crowding_distance += distance


def _select_tournament(population: Sequence[Individual], tournament_size: int = 3) -> Individual:
    """Performs tournament selection based on rank and crowding distance."""

    tournament_size = max(1, min(tournament_size, len(population)))
    contenders = random.sample(population, tournament_size)
    contenders.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))
    return contenders[0]


class NSGA2:
    """Implementation of the NSGA-II algorithm for hyperparameter optimisation."""

    def __init__(
        self,
        datasets: Sequence[DatasetInfo],
        population_size: int = 50,
        num_generations: int = 100,
        seed: int = 42,
        output_dir: str = "results",
    ) -> None:
        self.datasets = datasets
        if population_size != 50:
            print(
                f"Requested population size {population_size} but using 50 per specification.",
                flush=True,
            )
        self.population_size = 50
        self.num_generations = min(num_generations, 100)
        if num_generations > 100:
            print(
                f"Requested {num_generations} generations but limiting to 100 per specification.",
                flush=True,
            )
        self.random = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = EvaluationLogger(output_dir, datasets)
        self.evaluator = Evaluator(datasets, self.device)

    def _initial_population(self) -> List[Individual]:
        return [Individual(genome=Genome.random()) for _ in range(self.population_size)]

    def _evaluate_population(
        self,
        population: Iterable[Individual],
        generation: int,
    ) -> None:
        if not isinstance(population, list):
            population = list(population)
        total_individuals = len(population)
        for index, individual in enumerate(population):
            if individual.fitness is None:
                metrics_by_dataset, fold_metrics_by_dataset = self.evaluator.evaluate(
                    genome=individual.genome,
                    generation=generation,
                    individual_index=index,
                    total_individuals=total_individuals,
                )
                objectives = _compute_objectives(fold_metrics_by_dataset)
                individual.fitness = objectives
                individual.metrics_by_dataset = metrics_by_dataset
                individual.fold_metrics_by_dataset = fold_metrics_by_dataset

    def run(self) -> List[Individual]:
        """Executes the NSGA-II loop and returns the final population."""

        population = self._initial_population()
        self._evaluate_population(population, generation=0)

        initial_fronts = _non_dominated_sort(population)
        for front in initial_fronts:
            _compute_crowding_distance(front)
        if self.logger:
            self.logger.log_generation(0, population)

        for generation in range(1, self.num_generations + 1):
            offspring: List[Individual] = []
            while len(offspring) < self.population_size:
                parent1 = _select_tournament(population, tournament_size=3)
                parent2 = _select_tournament(population, tournament_size=3)
                child_genome = parent1.genome.crossover(parent2.genome).mutate(mutation_rate=0.1)
                offspring.append(Individual(genome=child_genome))

            self._evaluate_population(offspring, generation=generation)

            combined = population + offspring
            fronts = _non_dominated_sort(combined)
            new_population: List[Individual] = []
            for front in fronts:
                _compute_crowding_distance(front)
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    front_sorted = sorted(front, key=lambda ind: (ind.rank, -ind.crowding_distance))
                    remaining_slots = self.population_size - len(new_population)
                    new_population.extend(front_sorted[:remaining_slots])
                    break

            population = new_population

            updated_fronts = _non_dominated_sort(population)
            for front in updated_fronts:
                _compute_crowding_distance(front)
            if self.logger:
                self.logger.log_generation(generation, population)

        return population
