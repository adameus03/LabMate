import argparse
from typing import Literal

import numpy as np
import torch

from model import Model


def _build_sequences(
    inputs: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
):
    """
    Build sliding-window sequences from flat inputs/targets.

    Returns:
        seq_inputs:  (num_sequences, L, 416)
        seq_targets: (num_sequences, L, 1)
    """
    assert inputs.shape[0] == targets.shape[0], "Inputs and targets length mismatch"
    num_samples = inputs.shape[0]
    num_sequences = num_samples - sequence_length + 1
    if num_sequences <= 0:
        raise ValueError(
            f"Not enough samples ({num_samples}) for sequence_length={sequence_length}"
        )

    seq_inputs = []
    seq_targets = []
    for i in range(num_sequences):
        seq_inputs.append(inputs[i : i + sequence_length])
        seq_targets.append(targets[i : i + sequence_length])

    return (
        np.stack(seq_inputs, axis=0).astype(np.float32),
        np.stack(seq_targets, axis=0).astype(np.float32),
    )


def evaluate_model_continuous(
    model_path: str,
    use_val: bool = False,
    sequence_length: int = 100,
    device: str = "cuda",
    eval_mode: Literal["final", "all"] = "final",
):
    """
    Evaluate continuous 1D coordinate model on either validation or test data.

    Args:
        model_path: Path to checkpoint (.pth) produced by continuous training.
        use_val: If True, use validation data (1500*19) instead of test data (3000*19).
        sequence_length: Sliding-window sequence length (must match training).
        device: 'cuda' or 'cpu'.
        eval_mode: 'final' to evaluate only last timestep in each sequence,
                   'all' to use all timesteps.
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

    print(f"Total samples: {inputs.shape[0]}")
    feature_dim = inputs.shape[1]
    if feature_dim != 416:
        raise ValueError(f"Expected 416 input features, got {feature_dim}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Sequence length: {sequence_length}")
    print(f"Eval mode: {eval_mode}")

    seq_inputs, seq_targets = _build_sequences(inputs, targets, sequence_length)
    print(f"Number of sequences: {seq_inputs.shape[0]}")

    # Convert to torch
    seq_inputs_t = torch.FloatTensor(seq_inputs).to(device)
    seq_targets_t = torch.FloatTensor(seq_targets).to(device)

    # Prepare model
    print("\n=== MODEL LOADING ===")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

    all_preds = []
    all_trues = []

    with torch.no_grad():
        batch_size = 32
        num_sequences = seq_inputs_t.size(0)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, num_sequences)
            current_batch_size = end - start

            batch_inputs = seq_inputs_t[start:end]  # (B, L, 208)
            batch_targets = seq_targets_t[start:end]  # (B, L, 1)

            c1 = torch.zeros(current_batch_size, 200, device=device)
            c2 = torch.zeros(current_batch_size, 96, device=device)
            c3 = torch.zeros(current_batch_size, 1, device=device)

            h1 = torch.zeros(current_batch_size, 200, device=device)
            h2 = torch.zeros(current_batch_size, 96, device=device)
            h3 = torch.zeros(current_batch_size, 1, device=device)

            for t in range(sequence_length):
                c1, c2, c3, h1, h2, h3 = model(
                    batch_inputs[:, t, :],
                    (c1, c2, c3),
                    (h1, h2, h3),
                )

                if eval_mode == "all" or (eval_mode == "final" and t == sequence_length - 1):
                    preds_step = h3.squeeze(-1).cpu().numpy()  # (B,)
                    trues_step = batch_targets[:, t, 0].cpu().numpy()  # (B,)
                    all_preds.extend(preds_step.tolist())
                    all_trues.extend(trues_step.tolist())

            if (b + 1) % 50 == 0 or b == num_batches - 1:
                print(f"Processed {b + 1}/{num_batches} batches")

    all_preds_np = np.array(all_preds, dtype=np.float32)
    all_trues_np = np.array(all_trues, dtype=np.float32)

    print("\n=== REGRESSION METRICS ===")
    mse = np.mean((all_preds_np - all_trues_np) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(all_preds_np - all_trues_np)))

    print(f"Num predictions: {all_preds_np.shape[0]}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    # Basic distribution info
    print("\n=== PREDICTION / TARGET RANGES ===")
    print(f"Targets:  min={float(all_trues_np.min()):.4f}, max={float(all_trues_np.max()):.4f}")
    print(f"Preds:    min={float(all_preds_np.min()):.4f}, max={float(all_preds_np.max()):.4f}")

    return mse, rmse, mae


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continuous 1D location LSTM model"
    )
    parser.add_argument("model_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument(
        "--use_val",
        action="store_true",
        help="Use validation data instead of test data for evaluation",
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
        help="Use only final timestep or all timesteps for evaluation",
    )

    args = parser.parse_args()

    evaluate_model_continuous(
        model_path=args.model_path,
        use_val=args.use_val,
        sequence_length=args.sequence_length,
        device=args.device,
        eval_mode=args.eval_mode,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()

