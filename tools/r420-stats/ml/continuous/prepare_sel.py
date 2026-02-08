import os
from typing import List, Tuple, Dict
import re

import numpy as np
import pandas as pd


def _load_single_csv(
    csv_path: str,
    max_rows: int = None,
) -> np.ndarray:
    """
    Load rows from a single CSV as float32.

    Assumes no header row and a fixed number of feature columns (expected 416).
    If max_rows is None, loads all rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, nrows=max_rows)
    return df.values.astype(np.float32)


def _extract_file_index(csv_path: str) -> int:
    """
    Extract the numeric index from a t*.csv filename.
    
    For example: t1.csv -> 1, t15.csv -> 15
    """
    basename = os.path.basename(csv_path)
    match = re.match(r"^t(\d+)\.csv$", basename)
    if not match:
        raise ValueError(f"Filename {basename} doesn't match pattern t*.csv")
    return int(match.group(1))


def _build_coordinate_for_index(idx: int, min_idx: int, max_idx: int) -> float:
    """
    Map file index linearly to [0.0, 1.0] based on min and max indices.

    This represents the 1D location coordinate associated with the data
    in t{idx}.csv along the measurement path.
    """
    if max_idx == min_idx:
        return 0.0
    return (idx - min_idx) / (max_idx - min_idx)


def create_datasets_from_file_lists(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    max_rows_per_file: int = None,
) -> Tuple[str, str, str, str, str, str]:
    """
    Create train/val/test datasets from separate lists of CSV files.

    Each file is loaded completely (up to max_rows_per_file if specified)
    and assigned to its designated split (train/val/test).
    
    Coordinates are assigned based on the file index (t1, t2, etc.) to ensure
    monotonicity with respect to the original measurement sequence.

    Args:
        train_files: List of CSV file paths for training set
        val_files: List of CSV file paths for validation set
        test_files: List of CSV file paths for test set
        max_rows_per_file: Maximum rows to load per file (None = all rows)

    Returns:
        Paths of created .npy files in the order:
            (train_inputs, train_targets, val_inputs, val_targets,
             test_inputs, test_targets)
    """
    # Combine all files and extract their indices
    all_files = train_files + val_files + test_files
    
    # Extract file indices to determine coordinate range
    file_indices = [_extract_file_index(f) for f in all_files]
    min_idx = min(file_indices)
    max_idx = max(file_indices)
    
    # Create mapping from file path to coordinate based on actual file index
    file_to_coord = {}
    for file_path in all_files:
        idx = _extract_file_index(file_path)
        coord = _build_coordinate_for_index(idx, min_idx, max_idx)
        file_to_coord[file_path] = coord

    expected_feature_dim = 416

    all_train_inputs: List[np.ndarray] = []
    all_train_targets: List[np.ndarray] = []

    all_val_inputs: List[np.ndarray] = []
    all_val_targets: List[np.ndarray] = []

    all_test_inputs: List[np.ndarray] = []
    all_test_targets: List[np.ndarray] = []

    # Process training files
    print(f"\nProcessing {len(train_files)} training files:")
    for csv_path in train_files:
        file_idx = _extract_file_index(csv_path)
        print(f"  Loading {csv_path} (index={file_idx})")
        data = _load_single_csv(csv_path, max_rows=max_rows_per_file)

        # Enforce consistent feature dimensionality
        feat_dim = data.shape[1]
        if feat_dim != expected_feature_dim:
            raise ValueError(
                f"File {csv_path} has {feat_dim} features per row, "
                f"but {expected_feature_dim} are required."
            )

        coord = file_to_coord[csv_path]
        coord_array = np.full((data.shape[0], 1), coord, dtype=np.float32)

        all_train_inputs.append(data)
        all_train_targets.append(coord_array)
        print(f"    coord={coord:.4f}, rows={data.shape[0]}")

    # Process validation files
    print(f"\nProcessing {len(val_files)} validation files:")
    for csv_path in val_files:
        file_idx = _extract_file_index(csv_path)
        print(f"  Loading {csv_path} (index={file_idx})")
        data = _load_single_csv(csv_path, max_rows=max_rows_per_file)

        feat_dim = data.shape[1]
        if feat_dim != expected_feature_dim:
            raise ValueError(
                f"File {csv_path} has {feat_dim} features per row, "
                f"but {expected_feature_dim} are required."
            )

        coord = file_to_coord[csv_path]
        coord_array = np.full((data.shape[0], 1), coord, dtype=np.float32)

        all_val_inputs.append(data)
        all_val_targets.append(coord_array)
        print(f"    coord={coord:.4f}, rows={data.shape[0]}")

    # Process test files
    print(f"\nProcessing {len(test_files)} test files:")
    for csv_path in test_files:
        file_idx = _extract_file_index(csv_path)
        print(f"  Loading {csv_path} (index={file_idx})")
        data = _load_single_csv(csv_path, max_rows=max_rows_per_file)

        feat_dim = data.shape[1]
        if feat_dim != expected_feature_dim:
            raise ValueError(
                f"File {csv_path} has {feat_dim} features per row, "
                f"but {expected_feature_dim} are required."
            )

        coord = file_to_coord[csv_path]
        coord_array = np.full((data.shape[0], 1), coord, dtype=np.float32)

        all_test_inputs.append(data)
        all_test_targets.append(coord_array)
        print(f"    coord={coord:.4f}, rows={data.shape[0]}")

    # Concatenate all splits
    train_inputs_np = np.concatenate(all_train_inputs, axis=0) if all_train_inputs else np.array([]).reshape(0, expected_feature_dim)
    train_targets_np = np.concatenate(all_train_targets, axis=0) if all_train_targets else np.array([]).reshape(0, 1)

    val_inputs_np = np.concatenate(all_val_inputs, axis=0) if all_val_inputs else np.array([]).reshape(0, expected_feature_dim)
    val_targets_np = np.concatenate(all_val_targets, axis=0) if all_val_targets else np.array([]).reshape(0, 1)

    test_inputs_np = np.concatenate(all_test_inputs, axis=0) if all_test_inputs else np.array([]).reshape(0, expected_feature_dim)
    test_targets_np = np.concatenate(all_test_targets, axis=0) if all_test_targets else np.array([]).reshape(0, 1)

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
        description="Prepare train/val/test datasets by assigning complete CSV files to splits"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="Directory containing t*.csv (defaults to tools/r420-stats relative to this script)",
    )
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        help="List of file numbers for training (e.g., 1 2 3 4 5)",
    )
    parser.add_argument(
        "--val_files",
        type=str,
        nargs="+",
        help="List of file numbers for validation (e.g., 6 7)",
    )
    parser.add_argument(
        "--test_files",
        type=str,
        nargs="+",
        help="List of file numbers for testing (e.g., 8 9 10)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum rows to load per file (default: load all rows)",
    )
    args = parser.parse_args()

    # Determine CSV directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = (
        args.csv_dir
        if args.csv_dir is not None
        else os.path.dirname(os.path.dirname(script_dir))
    )

    # Build file paths from file numbers
    def build_file_paths(file_numbers, csv_dir):
        if file_numbers is None:
            return []
        paths = []
        for num in file_numbers:
            file_path = os.path.join(csv_dir, f"t{num}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            paths.append(file_path)
        return paths

    train_files = build_file_paths(args.train_files, csv_dir)
    val_files = build_file_paths(args.val_files, csv_dir)
    test_files = build_file_paths(args.test_files, csv_dir)

    if not train_files and not val_files and not test_files:
        raise ValueError(
            "No files specified! Use --train_files, --val_files, and/or --test_files"
        )

    print("Dataset configuration:")
    print(f"  Training files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
    print(f"  Validation files ({len(val_files)}): {[os.path.basename(f) for f in val_files]}")
    print(f"  Test files ({len(test_files)}): {[os.path.basename(f) for f in test_files]}")

    create_datasets_from_file_lists(train_files, val_files, test_files, max_rows_per_file=args.max_rows)
