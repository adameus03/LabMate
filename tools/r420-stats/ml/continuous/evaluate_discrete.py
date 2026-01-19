import argparse
from typing import Literal

import numpy as np
import torch

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

    total_correct = 0
    total_samples = 0

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

                    snapped_preds, _ = _nearest_level(preds_step, levels)
                    snapped_trues, _ = _nearest_level(trues_step, levels)

                    total_correct += np.sum(snapped_preds == snapped_trues)
                    total_samples += preds_step.shape[0]

                    all_preds.append(preds_step)
                    all_trues.append(trues_step)

            if (b + 1) % 50 == 0 or b == num_batches - 1:
                print(f"Processed {b + 1}/{num_batches} batches")

    # Regression metrics for reference
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_trues_np = np.concatenate(all_trues, axis=0)

    mse = float(np.mean((all_preds_np - all_trues_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(all_preds_np - all_trues_np)))

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print("\n=== DISCRETE ACCURACY (snapped to nearest level) ===")
    print(f"Levels detected: {num_levels}")
    print(f"Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")

    print("\n=== REGRESSION METRICS (before snapping) ===")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    return accuracy, mse, rmse, mae


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

