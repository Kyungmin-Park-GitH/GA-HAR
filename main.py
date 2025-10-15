"""Entry point for running the NSGA-II optimisation on the HAR datasets."""
from __future__ import annotations

import argparse
from typing import List

from src.data_utils import load_dataset
from src.ga import NSGA2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise HAR models with NSGA-II")
    parser.add_argument(
        "--coventry-path",
        default="./data/coventry_2018/40_linear_sensor1",
        help="Path to the Coventry dataset directory containing CSV files.",
    )
    parser.add_argument(
        "--coventry-test-indices",
        default="diaz_coventry_108.txt",
        help="Path to the test index file for the Coventry dataset.",
    )
    parser.add_argument(
        "--infra-path",
        default="./data/infra_adl2018/40_sensor3",
        help="Path to the INFRA dataset directory containing CSV files.",
    )
    parser.add_argument(
        "--infra-test-indices",
        default="diaz_infra_122.txt",
        help="Path to the test index file for the INFRA dataset.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations to run the GA for.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=30,
        help="Number of individuals per generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where evaluation CSV logs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    coventry = load_dataset(
        dataset_path=args.coventry_path,
        test_index_path=args.coventry_test_indices,
        name="coventry",
    )
    infra = load_dataset(
        dataset_path=args.infra_path,
        test_index_path=args.infra_test_indices,
        name="infra",
    )

    optimiser = NSGA2(
        datasets=[coventry, infra],
        population_size=args.population_size,
        num_generations=args.generations,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    final_population = optimiser.run()

    output_lines: List[str] = []
    output_lines.append("Final Pareto front (approximate):")
    for individual in final_population:
        if individual.rank == 0:
            genome = individual.genome
            acc_coventry, acc_infra = individual.fitness if individual.fitness else (0.0, 0.0)
            output_lines.append(
                (
                    f"Accuracy Coventry: {acc_coventry:.4f}, Accuracy Infra: {acc_infra:.4f}, "
                    f"Genome: {genome}"
                )
            )

    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
