"""
Calculate best performing sample indices from find_best_num.py results.
Selects top performers within each label group based on True ratio (correct_count / test_count).
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd


def load_result_file(file_path: str) -> pd.DataFrame:
    """Load result CSV file."""
    df = pd.read_csv(file_path)
    df.columns = ["index", "label", "test_count", "correct_count"]
    return df


def calculate_true_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate True ratio (correct_count / test_count)."""
    df = df.copy()
    # Avoid division by zero
    df["true_ratio"] = np.where(
        df["test_count"] > 0,
        df["correct_count"] / df["test_count"],
        0.0
    )
    return df


def select_top_indices_by_label(
    df: pd.DataFrame,
    top_counts: Dict[int, int]
) -> List[int]:
    """
    Select top performing indices for each label.

    Args:
        df: DataFrame with columns [index, label, test_count, correct_count, true_ratio]
        top_counts: Dictionary mapping label -> number of top indices to select

    Returns:
        List of selected indices (1-based)
    """
    selected_indices: List[int] = []

    for label, count in top_counts.items():
        # Filter by label
        label_df = df[df["label"] == label].copy()

        if len(label_df) == 0:
            print(f"Warning: No samples found for label {label}")
            continue

        # Sort by true_ratio descending, then by index ascending (for ties)
        label_df = label_df.sort_values(
            by=["true_ratio", "index"],
            ascending=[False, True]
        )

        # Select top n
        top_n = min(count, len(label_df))
        top_indices = label_df.head(top_n)["index"].tolist()
        selected_indices.extend(top_indices)

        print(f"Label {label}: selected {top_n} indices (requested {count})")

    return selected_indices


def save_indices_to_file(indices: List[int], output_path: str) -> None:
    """Save indices to text file, one per line, sorted."""
    indices_sorted = sorted(indices)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in indices_sorted:
            f.write(f"{idx}\n")
    print(f"Saved {len(indices_sorted)} indices to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate best performing indices from find_best_num.py results"
    )
    parser.add_argument(
        "--input-dir",
        default="./results",
        help="Directory containing coventry_result.csv and infra_result.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to save output text files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Define top counts for each dataset and label
    # =========================================================================

    # Coventry: label 5 -> 18, all others -> 9
    coventry_top_counts: Dict[int, int] = {}
    for label in range(11):  # labels 0-10
        if label == 5:
            coventry_top_counts[label] = 18
        else:
            coventry_top_counts[label] = 9

    # Infra:
    # labels 3,4 -> 16
    # labels 5,8,9 -> 15
    # labels 0,1,2 -> 8
    # labels 6,7,10 -> 7
    infra_top_counts: Dict[int, int] = {
        0: 8,
        1: 8,
        2: 8,
        3: 16,
        4: 16,
        5: 15,
        6: 7,
        7: 7,
        8: 15,
        9: 15,
        10: 7,
    }

    # =========================================================================
    # Process Coventry
    # =========================================================================
    print("=" * 60)
    print("Processing Coventry")
    print("=" * 60)

    coventry_input = os.path.join(args.input_dir, "coventry_result.csv")
    if os.path.exists(coventry_input):
        cov_df = load_result_file(coventry_input)
        cov_df = calculate_true_ratio(cov_df)

        cov_selected = select_top_indices_by_label(cov_df, coventry_top_counts)

        cov_output = os.path.join(args.output_dir, "coventry_best_indices.txt")
        save_indices_to_file(cov_selected, cov_output)
    else:
        print(f"Error: {coventry_input} not found")

    # =========================================================================
    # Process Infra
    # =========================================================================
    print("\n" + "=" * 60)
    print("Processing Infra")
    print("=" * 60)

    infra_input = os.path.join(args.input_dir, "infra_result.csv")
    if os.path.exists(infra_input):
        infra_df = load_result_file(infra_input)
        infra_df = calculate_true_ratio(infra_df)

        infra_selected = select_top_indices_by_label(infra_df, infra_top_counts)

        infra_output = os.path.join(args.output_dir, "infra_best_indices.txt")
        save_indices_to_file(infra_selected, infra_output)
    else:
        print(f"Error: {infra_input} not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
