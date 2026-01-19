import os
from typing import List, Tuple
import re

import numpy as np
import pandas as pd


def _load_single_csv(
    csv_path: str,
    max_rows: int = 15000,
) -> np.ndarray:
    """
    Load up to `max_rows` rows from a single CSV as float32.

    Assumes no header row and a fixed number of feature columns (expected 416).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, nrows=max_rows)
    return df.values.astype(np.float32)


def _build_coordinate_for_index(idx: int, total: int) -> float:
    """
    Map file index (0..total-1) linearly to [0.0, 1.0].

    This represents the 1D location coordinate associated with the data
    in t{idx+1}.csv along the measurement path.
    """
    if total <= 1:
        return 0.0
    return idx / (total - 1)


def create_datasets_from_t_csvs(
    csv_files: List[str],
    max_rows_per_file: int = 15000,
    train_count: int = 10500,
    val_count: int = 1500,
    test_count: int = 3000,
) -> Tuple[str, str, str, str, str, str]:
    """
    Create continuous train/val/test datasets from t1..tN.csv files.

    For each CSV:
        - Use first `max_rows_per_file` rows
        - Split into:
            train: first `train_count`
            val:   next  `val_count`
            test:  next  `test_count`
        - Assign a scalar coordinate in [0, 1] based on file index.

    Returns:
        Paths of created .npy files in the order:
            (train_inputs, train_targets, val_inputs, val_targets,
             test_inputs, test_targets)
    """
    num_files = len(csv_files)

    expected_feature_dim = 416

    all_train_inputs: List[np.ndarray] = []
    all_train_targets: List[np.ndarray] = []

    all_val_inputs: List[np.ndarray] = []
    all_val_targets: List[np.ndarray] = []

    all_test_inputs: List[np.ndarray] = []
    all_test_targets: List[np.ndarray] = []

    for idx, csv_path in enumerate(csv_files):
        print(f"Processing {csv_path} ({idx + 1}/{num_files})")
        data = _load_single_csv(csv_path, max_rows=max_rows_per_file)

        # Enforce consistent feature dimensionality of 416
        feat_dim = data.shape[1]
        if feat_dim != expected_feature_dim:
            raise ValueError(
                f"File {csv_path} has {feat_dim} features per row, "
                f"but {expected_feature_dim} are required to match the model input."
            )

        if data.shape[0] < train_count + val_count + test_count:
            raise ValueError(
                f"File {csv_path} has only {data.shape[0]} rows, "
                f"but requires at least {train_count + val_count + test_count}."
            )

        coord = _build_coordinate_for_index(idx, num_files)
        coord_array = np.full((data.shape[0], 1), coord, dtype=np.float32)

        # Train / val / test splits by consecutive rows
        train_inputs = data[:train_count]
        train_targets = coord_array[:train_count]

        val_start = train_count
        val_end = train_count + val_count
        val_inputs = data[val_start:val_end]
        val_targets = coord_array[val_start:val_end]

        test_start = val_end
        test_end = val_end + test_count
        test_inputs = data[test_start:test_end]
        test_targets = coord_array[test_start:test_end]

        all_train_inputs.append(train_inputs)
        all_train_targets.append(train_targets)

        all_val_inputs.append(val_inputs)
        all_val_targets.append(val_targets)

        all_test_inputs.append(test_inputs)
        all_test_targets.append(test_targets)

        print(
            f"  coord={coord:.4f}, "
            f"train={train_inputs.shape[0]}, "
            f"val={val_inputs.shape[0]}, "
            f"test={test_inputs.shape[0]}"
        )

    # Concatenate across all locations
    train_inputs_np = np.concatenate(all_train_inputs, axis=0)
    train_targets_np = np.concatenate(all_train_targets, axis=0)

    val_inputs_np = np.concatenate(all_val_inputs, axis=0)
    val_targets_np = np.concatenate(all_val_targets, axis=0)

    test_inputs_np = np.concatenate(all_test_inputs, axis=0)
    test_targets_np = np.concatenate(all_test_targets, axis=0)

    print("\nFinal dataset sizes:")
    print(f"  Train: {train_inputs_np.shape} (inputs), {train_targets_np.shape} (targets)")
    print(f"  Val:   {val_inputs_np.shape} (inputs), {val_targets_np.shape} (targets)")
    print(f"  Test:  {test_inputs_np.shape} (inputs), {test_targets_np.shape} (targets)")

    # Save .npy files in the current directory
    train_inputs_path = "train_inputs.npy"
    train_targets_path = "train_targets.npy"
    val_inputs_path = "val_inputs.npy"
    val_targets_path = "val_targets.npy"
    test_inputs_path = "test_inputs.npy"
    test_targets_path = "test_targets.npy"

    np.save(train_inputs_path, train_inputs_np)
    np.save(train_targets_path, train_targets_np)
    np.save(val_inputs_path, val_inputs_np)
    np.save(val_targets_path, val_targets_np)
    np.save(test_inputs_path, test_inputs_np)
    np.save(test_targets_path, test_targets_np)

    print("\nSaved .npy files:")
    print(f"  {train_inputs_path}, {train_targets_path}")
    print(f"  {val_inputs_path}, {val_targets_path}")
    print(f"  {test_inputs_path}, {test_targets_path}")

    return (
        train_inputs_path,
        train_targets_path,
        val_inputs_path,
        val_targets_path,
        test_inputs_path,
        test_targets_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare continuous train/val/test datasets from tools/r420-stats/t*.csv"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="Directory containing t*.csv (defaults to tools/r420-stats relative to this script)",
    )
    args = parser.parse_args()

    # Discover t1..tN.csv in tools/r420-stats directory
    # __file__ is tools/r420-stats/ml/continuous/prepare.py
    # We need to go up 2 levels: continuous -> ml -> r420-stats
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = (
        args.csv_dir
        if args.csv_dir is not None
        else os.path.dirname(os.path.dirname(script_dir))
    )

    t_csv_re = re.compile(r"^t(\d+)\.csv$")
    numbered: List[Tuple[int, str]] = []

    for fname in os.listdir(csv_dir):
        m = t_csv_re.match(fname)
        if m is None:
            continue
        n = int(m.group(1))
        numbered.append((n, os.path.join(csv_dir, fname)))

    # Sort by the numeric index so coordinates map consistently t1..tN
    numbered.sort(key=lambda x: x[0])
    csv_files = [path for _, path in numbered]

    if not csv_files:
        raise RuntimeError(f"No t*.csv files found in {csv_dir}")

    print("Found CSV files:")
    for p in csv_files:
        print(f"  {p}")

    create_datasets_from_t_csvs(csv_files)

