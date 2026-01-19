import argparse
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from model import Model


def _nearest_level(pred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Map each prediction to the nearest discrete coordinate level.
    """
    # levels shape: (K,)
    # pred shape: (...,)
    # Compute abs diff to all levels and take argmin
    diffs = np.abs(pred[..., None] - levels[None, ...])
    idx = np.argmin(diffs, axis=-1)
    return levels[idx], idx


def _extract_levels_from_targets(targets: np.ndarray) -> np.ndarray:
    """
    Derive unique coordinate levels from target values (flattened) and sort them.
    A small tolerance is used to merge floating point noise.
    """
    flat = targets.reshape(-1)
    # Round to 6 decimals to group near-identical levels
    rounded = np.round(flat, 6)
    levels = np.unique(rounded)
    return levels


def evaluate_discrete_accuracy(
    model_path: str,
    use_val: bool = False,
    sequence_length: int = 100,
    device: str = "cuda",
    eval_mode: Literal["final", "all"] = "final",
):
    """
    Evaluate the continuous model, but report accuracy after snapping predictions
    to the nearest discrete location level derived from the targets.
    """
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

    # Load data
    if use_val:
        print("Loading VALIDATION data for evaluation")
        inputs = np.load("val_inputs.npy")
        targets = np.load("val_targets.npy")
    else:
        print("Loading TEST data for evaluation")
        inputs = np.load("test_inputs.npy")
        targets = np.load("test_targets.npy")

    feature_dim = inputs.shape[1]
    if feature_dim != 416:
        raise ValueError(f"Expected 416 input features, got {feature_dim}")

    print(f"Total samples: {inputs.shape[0]}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Sequence length: {sequence_length}")
    print(f"Eval mode: {eval_mode}")

    levels = _extract_levels_from_targets(targets)
    num_levels = len(levels)
    print(f"Detected {num_levels} location levels from targets")

    # Build sequences
    num_samples = inputs.shape[0]
    num_sequences = num_samples - sequence_length + 1
    if num_sequences <= 0:
        raise ValueError(
            f"Not enough samples ({num_samples}) for sequence_length={sequence_length}"
        )
    print(f"Number of sequences: {num_sequences}")

    # Prepare model
    print("\n=== MODEL LOADING ===")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

    # Collect discrete class predictions and true labels (class indices 0..num_levels-1)
    all_predicted_classes = []
    all_true_classes = []

    # For reference, also accumulate regression errors before snapping
    all_preds = []
    all_trues = []

    with torch.no_grad():
        batch_size = 32
        num_batches = (num_sequences + batch_size - 1) // batch_size

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, num_sequences)
            current_batch_size = end - start

            batch_inputs = np.stack(
                [inputs[i : i + sequence_length] for i in range(start, end)], axis=0
            ).astype(np.float32)
            batch_targets = np.stack(
                [targets[i : i + sequence_length] for i in range(start, end)], axis=0
            ).astype(np.float32)

            batch_inputs_t = torch.FloatTensor(batch_inputs).to(device)
            batch_targets_t = torch.FloatTensor(batch_targets).to(device)

            c1 = torch.zeros(current_batch_size, 200, device=device)
            c2 = torch.zeros(current_batch_size, 96, device=device)
            c3 = torch.zeros(current_batch_size, 1, device=device)

            h1 = torch.zeros(current_batch_size, 200, device=device)
            h2 = torch.zeros(current_batch_size, 96, device=device)
            h3 = torch.zeros(current_batch_size, 1, device=device)

            for t in range(sequence_length):
                c1, c2, c3, h1, h2, h3 = model(
                    batch_inputs_t[:, t, :],
                    (c1, c2, c3),
                    (h1, h2, h3),
                )

                if eval_mode == "all" or (eval_mode == "final" and t == sequence_length - 1):
                    preds_step = h3.squeeze(-1).cpu().numpy()  # (B,)
                    trues_step = batch_targets_t[:, t, 0].cpu().numpy()  # (B,)

                    # Map to nearest level and get class index
                    _, pred_class_idx = _nearest_level(preds_step, levels)
                    _, true_class_idx = _nearest_level(trues_step, levels)

                    all_predicted_classes.extend(pred_class_idx)
                    all_true_classes.extend(true_class_idx)

                    all_preds.append(preds_step)
                    all_trues.append(trues_step)

            if (b + 1) % 50 == 0 or b == num_batches - 1:
                print(f"Processed {b + 1}/{num_batches} batches")

    # Convert to numpy arrays
    all_predicted_classes = np.array(all_predicted_classes)
    all_true_classes = np.array(all_true_classes)

    # Regression metrics for reference
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_trues_np = np.concatenate(all_trues, axis=0)

    mse = float(np.mean((all_preds_np - all_trues_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(all_preds_np - all_trues_np)))

    print(f"\n=== PREDICTION ANALYSIS ===")
    print(f"Total predictions: {len(all_predicted_classes)}")

    print(f"\nPredicted class distribution:")
    pred_unique, pred_counts = np.unique(all_predicted_classes, return_counts=True)
    for cls, count in zip(pred_unique, pred_counts):
        print(f"  Class {cls}: {count} predictions ({100*count/len(all_predicted_classes):.2f}%)")

    print(f"\nTrue class distribution:")
    true_unique, true_counts = np.unique(all_true_classes, return_counts=True)
    for cls, count in zip(true_unique, true_counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(all_true_classes):.2f}%)")

    # Check for model collapse
    if len(pred_unique) == 1:
        print("\n⚠️  WARNING: Model is predicting only ONE class!")
        print("This indicates a collapsed model.")

    # Calculate confusion matrix (19x19)
    class_labels = list(range(num_levels))
    cm = confusion_matrix(all_true_classes, all_predicted_classes, labels=class_labels)

    # Print confusion matrix (compact format for 19x19)
    print("\n=== CONFUSION MATRIX ===")
    print(f"Matrix size: {num_levels}x{num_levels}")
    print("True \\ Predicted", end="")
    # Print header with all class indices (compact 3-digit format)
    header_str = " ".join([f"{i:3d}" for i in class_labels])
    print(f"  {header_str}")
    print("-" * (len(header_str) + 20))

    # Print all rows
    for i in range(num_levels):
        row_str = " ".join([f"{cm[i, j]:3d}" for j in range(num_levels)])
        print(f"{i:2d} | {row_str}")

    # Calculate per-class metrics
    print("\n=== PER-CLASS METRICS ===")
    for i in range(num_levels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Class {i:2d} (coord={levels[i]:.4f}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    # Calculate overall accuracy
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    print(f"\n=== OVERALL ACCURACY ===")
    print(f"Accuracy: {accuracy:.4f} ({np.trace(cm)}/{cm.sum()} correct)")

    print("\n=== REGRESSION METRICS (before snapping) ===")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    return accuracy, mse, rmse, mae, cm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continuous model as discrete location classifier"
    )
    parser.add_argument("model_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument(
        "--use_val",
        action="store_true",
        help="Use validation data instead of test data",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Sequence length (must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="final",
        choices=["final", "all"],
        help="Use only final timestep or all timesteps",
    )

    args = parser.parse_args()

    evaluate_discrete_accuracy(
        model_path=args.model_path,
        use_val=args.use_val,
        sequence_length=args.sequence_length,
        device=args.device,
        eval_mode=args.eval_mode,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()

