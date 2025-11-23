"""Entry point for running the NSGA-II optimisation on the HAR datasets (3 sensors)."""
from __future__ import annotations

import argparse
from typing import List

from src.data_utils_3 import load_dataset_3sensors

try:
    from src.ga_3 import NSGA2
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) == "src.ga_3":
        raise ModuleNotFoundError(
            "`src.ga_3` 모듈을 찾을 수 없습니다. 사용하려는 아키텍처의 GA 파일을 "
            "`src/ga_3.py`로 이름을 바꾼 후 다시 실행해 주세요. 예: `ga_lstm_3.py` → `ga_3.py`."
        ) from exc
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise 3-sensor HAR models with NSGA-II")
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
        "--coventry-test-indices",
        default="diaz_coventry_108.txt",
        help="Path to the test index file for the Coventry dataset.",
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
        "--infra-test-indices",
        default="diaz_infra_122.txt",
        help="Path to the test index file for the INFRA dataset.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations to run the GA for (capped at 100).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Number of individuals per generation (forced to 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_3sensor",
        help="Directory where evaluation CSV logs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    coventry, infra = load_dataset_3sensors(
        coventry_sensor1_path=args.coventry_sensor1_path,
        coventry_sensor2_path=args.coventry_sensor2_path,
        coventry_sensor3_path=args.coventry_sensor3_path,
        infra_sensor2_path=args.infra_sensor2_path,
        infra_sensor3_path=args.infra_sensor3_path,
        infra_sensor4_path=args.infra_sensor4_path,
        coventry_test_indices=args.coventry_test_indices,
        infra_test_indices=args.infra_test_indices,
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
        if individual.rank == 0 and individual.fitness is not None:
            genome = individual.genome
            mean_acc, min_acc, mean_f1, min_f1 = individual.fitness
            output_lines.append(
                (
                    "Objectives -> "
                    f"mean_acc: {mean_acc:.4f}, min_acc: {min_acc:.4f}, "
                    f"mean_f1: {mean_f1:.4f}, min_f1: {min_f1:.4f}; "
                    f"Genome: {genome}"
                )
            )

    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
