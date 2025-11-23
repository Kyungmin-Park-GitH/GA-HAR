"""Multi-objective genetic algorithm (NSGA-II) for 3-sensor CNN HAR model search."""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .data_utils_3 import DatasetInfo
from .training_cnn_3 import EvaluationMetrics, TrainingConfig, evaluate_with_kfold


HYPERPARAMETER_SPACE: Dict[str, Sequence] = {
    "learning_rate": [i / 10000 for i in range(1, 101)],
    "batch_size": [8, 16, 32, 48, 64, 96, 128, 192, 256],
    "conv_layers": [1, 2, 3, 4, 5],
    "filters": [1, 2, 4, 8, 16, 24, 32, 48, 64, 128],
    "kernel_size": [1, 2, 3, 4, 5, 6, 7, 8],
    "dropout_rate": [i / 100 for i in range(0, 51)],
}


def _compute_objectives(fold_metrics_by_dataset: Dict[str, Tuple[EvaluationMetrics, ...]]) -> Tuple[float, float, float, float]:
    if not fold_metrics_by_dataset:
        return 0.0, 0.0, 0.0, 0.0
    metrics_sequences = list(fold_metrics_by_dataset.values())
    fold_count = len(metrics_sequences[0]) if metrics_sequences else 0
    if fold_count == 0:
        return 0.0, 0.0, 0.0, 0.0
    for metrics in metrics_sequences:
        if len(metrics) != fold_count:
            raise ValueError("All datasets must provide the same number of folds")

    mean_acc, min_acc, mean_f1, min_f1 = [], [], [], []
    for fold_index in range(fold_count):
        accs = [metrics[fold_index].accuracy for metrics in metrics_sequences]
        f1s = [metrics[fold_index].f1 for metrics in metrics_sequences]
        mean_acc.append(sum(accs) / len(accs))
        min_acc.append(min(accs))
        mean_f1.append(sum(f1s) / len(f1s))
        min_f1.append(min(f1s))
    n = float(fold_count)
    return sum(mean_acc)/n, sum(min_acc)/n, sum(mean_f1)/n, sum(min_f1)/n


def _slugify(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_") or "dataset"


def _random_value(key: str) -> object:
    return random.choice(HYPERPARAMETER_SPACE[key])


@dataclass(eq=True, frozen=True)
class Genome:
    learning_rate: float
    batch_size: int
    conv_layers: int
    filters: int
    kernel_size: int
    dropout_rate: float

    def crossover(self, other: "Genome") -> "Genome":
        params = {f: getattr(self if random.random() < 0.5 else other, f) for f in self.__dataclass_fields__}
        return Genome(**params)

    def mutate(self, mutation_rate: float = 0.1) -> "Genome":
        params = {}
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if random.random() < mutation_rate:
                cands = [x for x in HYPERPARAMETER_SPACE[f] if x != v]
                params[f] = random.choice(cands) if cands else v
            else:
                params[f] = v
        return Genome(**params)

    @staticmethod
    def random() -> "Genome":
        return Genome(**{k: _random_value(k) for k in HYPERPARAMETER_SPACE})


@dataclass
class Individual:
    genome: Genome
    fitness: Tuple[float, float, float, float] | None = None
    crowding_distance: float = 0.0
    rank: int = math.inf
    metrics_by_dataset: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    fold_metrics_by_dataset: Dict[str, Tuple[EvaluationMetrics, ...]] = field(default_factory=dict)


class EvaluationLogger:
    def __init__(self, output_dir: str, datasets: Sequence[DatasetInfo]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = list(datasets)
        self.dataset_slugs = {d.name: _slugify(d.name) for d in datasets}
        self.hyperparameter_fields = list(Genome.__dataclass_fields__.keys())
        self.detail_files = {d.name: self.output_dir / f"{self.dataset_slugs[d.name]}_details.csv" for d in datasets}
        self.confusion_files = {d.name: self.output_dir / f"{self.dataset_slugs[d.name]}_confusion.csv" for d in datasets}
        self.overall_file = self.output_dir / "overall_results.csv"
        for p in [*self.detail_files.values(), *self.confusion_files.values(), self.overall_file]:
            if p.exists(): p.unlink()

    @staticmethod
    def _write_header(path: Path, header: Sequence[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    @staticmethod
    def _append_row(path: Path, row: Sequence[object]) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def log_generation(self, generation: int, individuals: Sequence[Individual]) -> None:
        if not individuals: return
        sorted_inds = sorted(individuals, key=lambda i: (i.rank, -i.crowding_distance))
        for d in self.datasets:
            if not self.detail_files[d.name].exists():
                self._write_header(self.detail_files[d.name], ["generation", "individual", "fold", *self.hyperparameter_fields, "accuracy", "recall", "precision", "f1_score"])
            if not self.confusion_files[d.name].exists():
                self._write_header(self.confusion_files[d.name], ["generation", "individual", "fold", *[f"cm_{i}_{j}" for i in range(d.num_classes) for j in range(d.num_classes)]])
        if not self.overall_file.exists():
            h = ["generation", "individual", *self.hyperparameter_fields]
            for d in self.datasets:
                s = self.dataset_slugs[d.name]
                h.extend([f"{s}_accuracy", f"{s}_recall", f"{s}_precision", f"{s}_f1_score"])
            h.extend(["mean_accuracy_objective", "min_accuracy_objective", "mean_f1_objective", "min_f1_objective"])
            self._write_header(self.overall_file, h)
        for ri, ind in enumerate(sorted_inds, start=1):
            for d in self.datasets:
                fm = ind.fold_metrics_by_dataset.get(d.name)
                if not fm: continue
                for fi, m in enumerate(fm, start=1):
                    self._append_row(self.detail_files[d.name], [generation, ri, fi, *[getattr(ind.genome, f) for f in self.hyperparameter_fields], f"{m.accuracy:.6f}", f"{m.recall:.6f}", f"{m.precision:.6f}", f"{m.f1:.6f}"])
                    self._append_row(self.confusion_files[d.name], [generation, ri, fi, *m.confusion_matrix.astype(int).flatten().tolist()])
            row = [generation, ri, *[getattr(ind.genome, f) for f in self.hyperparameter_fields]]
            for d in self.datasets:
                m = ind.metrics_by_dataset.get(d.name)
                if m: row.extend([f"{m.accuracy:.6f}", f"{m.recall:.6f}", f"{m.precision:.6f}", f"{m.f1:.6f}"])
            row.extend([f"{v:.6f}" for v in _compute_objectives(ind.fold_metrics_by_dataset)])
            self._append_row(self.overall_file, row)


class Evaluator:
    def __init__(self, datasets: Sequence[DatasetInfo], device: torch.device) -> None:
        self.datasets = datasets
        self.device = device

    def evaluate(self, genome: Genome, generation: int, individual_index: int, total_individuals: int) -> Tuple[Dict[str, EvaluationMetrics], Dict[str, Tuple[EvaluationMetrics, ...]]]:
        config = TrainingConfig(learning_rate=genome.learning_rate, batch_size=genome.batch_size, conv_layers=genome.conv_layers, filters=genome.filters, kernel_size=genome.kernel_size, dropout_rate=genome.dropout_rate)
        train_inputs1 = self.datasets[0].train.samples_sensor1
        train_inputs2 = self.datasets[0].train.samples_sensor2
        train_inputs3 = self.datasets[0].train.samples_sensor3
        train_labels = self.datasets[0].train.labels
        test_sets = {d.name: (d.test.samples_sensor1, d.test.samples_sensor2, d.test.samples_sensor3, d.test.labels) for d in self.datasets}
        def progress_callback(event: str, fold_index: int, dataset_name: str | None, metrics: EvaluationMetrics | None) -> None:
            prefix = f"[Generation {generation:03d}] Individual {individual_index + 1:02d}/{total_individuals:02d} Fold {fold_index + 1}/5"
            if event == "start": print(f"{prefix}: training...", flush=True)
            elif dataset_name and metrics: print(f"{prefix} test on '{dataset_name}': acc={metrics.accuracy:.4f}, recall={metrics.recall:.4f}, precision={metrics.precision:.4f}, f1={metrics.f1:.4f}", flush=True)
        num_classes = max(d.num_classes for d in self.datasets)
        fold_metrics_by_dataset = evaluate_with_kfold(config=config, inputs1=train_inputs1, inputs2=train_inputs2, inputs3=train_inputs3, labels=train_labels, test_sets=test_sets, num_classes=num_classes, device=self.device, folds=5, progress_callback=progress_callback)
        metrics_by_dataset = {d.name: EvaluationMetrics.average(fold_metrics_by_dataset[d.name]) for d in self.datasets if fold_metrics_by_dataset.get(d.name)}
        return metrics_by_dataset, fold_metrics_by_dataset


def _non_dominated_sort(population: Sequence[Individual]) -> List[List[Individual]]:
    fronts: List[List[Individual]] = []
    dom_counts = {i: 0 for i in range(len(population))}
    dominated = {i: [] for i in range(len(population))}
    for i, ind in enumerate(population):
        for j, other in enumerate(population):
            if i == j: continue
            if dominates(ind, other): dominated[i].append(j)
            elif dominates(other, ind): dom_counts[i] += 1
    current = [i for i, c in dom_counts.items() if c == 0]
    fi = 0
    while current:
        for idx in current: population[idx].rank = fi
        fronts.append([population[idx] for idx in current])
        nxt = []
        for idx in current:
            for di in dominated[idx]:
                dom_counts[di] -= 1
                if dom_counts[di] == 0: nxt.append(di)
        fi += 1
        current = nxt
    return fronts


def dominates(a: Individual, b: Individual) -> bool:
    if a.fitness is None or b.fitness is None: raise ValueError("Fitness required")
    return all(x >= y for x, y in zip(a.fitness, b.fitness)) and any(x > y for x, y in zip(a.fitness, b.fitness))


def _compute_crowding_distance(front: Sequence[Individual]) -> None:
    if not front: return
    num_obj = len(front[0].fitness or [])
    for ind in front: ind.crowding_distance = 0.0
    for obj in range(num_obj):
        fs = sorted(front, key=lambda i: i.fitness[obj] if i.fitness else 0.0)
        fs[0].crowding_distance = fs[-1].crowding_distance = float("inf")
        minv, maxv = fs[0].fitness[obj], fs[-1].fitness[obj]
        if math.isclose(maxv, minv): continue
        for i in range(1, len(fs) - 1):
            fs[i].crowding_distance += (fs[i+1].fitness[obj] - fs[i-1].fitness[obj]) / (maxv - minv)


def _select_tournament(population: Sequence[Individual], tournament_size: int = 3) -> Individual:
    ts = max(1, min(tournament_size, len(population)))
    contenders = sorted(random.sample(population, ts), key=lambda i: (i.rank, -i.crowding_distance))
    return contenders[0]


class NSGA2:
    def __init__(self, datasets: Sequence[DatasetInfo], population_size: int = 50, num_generations: int = 100, seed: int = 42, output_dir: str = "results") -> None:
        self.datasets = datasets
        self.population_size = 50
        self.num_generations = min(num_generations, 100)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = EvaluationLogger(output_dir, datasets)
        self.evaluator = Evaluator(datasets, self.device)

    def _initial_population(self) -> List[Individual]:
        return [Individual(genome=Genome.random()) for _ in range(self.population_size)]

    def _evaluate_population(self, population: Iterable[Individual], generation: int) -> None:
        pop = list(population)
        for idx, ind in enumerate(pop):
            if ind.fitness is None:
                mbd, fmbd = self.evaluator.evaluate(ind.genome, generation, idx, len(pop))
                ind.fitness = _compute_objectives(fmbd)
                ind.metrics_by_dataset = mbd
                ind.fold_metrics_by_dataset = fmbd

    def run(self) -> List[Individual]:
        population = self._initial_population()
        self._evaluate_population(population, 0)
        for f in _non_dominated_sort(population): _compute_crowding_distance(f)
        self.logger.log_generation(0, population)
        for gen in range(1, self.num_generations + 1):
            offspring = []
            while len(offspring) < self.population_size:
                p1, p2 = _select_tournament(population), _select_tournament(population)
                offspring.append(Individual(genome=p1.genome.crossover(p2.genome).mutate()))
            self._evaluate_population(offspring, gen)
            combined = population + offspring
            fronts = _non_dominated_sort(combined)
            new_pop = []
            for f in fronts:
                _compute_crowding_distance(f)
                if len(new_pop) + len(f) <= self.population_size: new_pop.extend(f)
                else:
                    new_pop.extend(sorted(f, key=lambda i: (i.rank, -i.crowding_distance))[:self.population_size - len(new_pop)])
                    break
            population = new_pop
            for f in _non_dominated_sort(population): _compute_crowding_distance(f)
            self.logger.log_generation(gen, population)
        return population
