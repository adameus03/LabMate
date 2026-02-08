import argparse
from typing import Literal, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

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


def _create_visualizations(
    all_preds: np.ndarray,
    all_trues: np.ndarray,
    mse: float,
    rmse: float,
    mae: float,
    output_dir: str = ".",
):
    """
    Create comprehensive evaluation plots for 1D RFID localization.

    Args:
        all_preds: Predicted coordinates (N,)
        all_trues: Ground truth coordinates (N,)
        mse: Mean squared error
        rmse: Root mean squared error
        mae: Mean absolute error
        output_dir: Directory to save plots
    """
    errors = all_preds - all_trues

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ========== 1. Predicted vs Actual (Main Plot) ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Scatter plot with color gradient based on error magnitude
    error_magnitude = np.abs(errors)
    scatter = ax1.scatter(
        all_trues, all_preds,
        c=error_magnitude,
        cmap='RdYlGn_r',
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.3
    )
    
    # Perfect prediction line
    min_val = min(all_trues.min(), all_preds.min())
    max_val = max(all_trues.max(), all_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Ground Truth Location (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Location (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Predicted vs Actual 1D Location', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Absolute Error (m)', fontsize=11, fontweight='bold')

    # ========== 2. Error Distribution (Histogram) ==========
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.4f}')
    ax2.set_xlabel('Prediction Error (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # ========== 3. Absolute Error Distribution ==========
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(error_magnitude, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(x=mae, color='darkred', linestyle='--', linewidth=2, label=f'MAE: {mae:.4f}')
    ax3.set_xlabel('Absolute Error (m)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ========== 4. Temporal Error Pattern ==========
    ax4 = fig.add_subplot(gs[2, 0])
    sample_size = min(500, len(errors))  # Plot first 500 for clarity
    ax4.plot(range(sample_size), errors[:sample_size], marker='o', markersize=3, linestyle='-', alpha=0.6, color='purple')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.fill_between(range(sample_size), -mae, mae, alpha=0.2, color='green', label='±MAE band')
    ax4.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Prediction Error (m)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Temporal Error Pattern (First {sample_size} Samples)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ========== 5. Metrics Summary Box ==========
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate additional statistics
    percentile_95 = np.percentile(error_magnitude, 95)
    percentile_90 = np.percentile(error_magnitude, 90)
    
    metrics_text = (
        f"REGRESSION METRICS\n"
        f"{'='*35}\n"
        f"Number of Predictions: {len(all_preds)}\n"
        f"MSE:  {mse:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAE:  {mae:.6f}\n"
        f"\nADDITIONAL STATISTICS\n"
        f"{'='*35}\n"
        f"Mean Error: {np.mean(errors):.6f}\n"
        f"Std Dev Error: {np.std(errors):.6f}\n"
        f"Min Error: {np.min(errors):.6f}\n"
        f"Max Error: {np.max(errors):.6f}\n"
        f"90th Percentile Error: {percentile_90:.6f}\n"
        f"95th Percentile Error: {percentile_95:.6f}\n"
        f"\nPREDICTION RANGES\n"
        f"{'='*35}\n"
        f"Target  - Min: {all_trues.min():.4f}, Max: {all_trues.max():.4f}\n"
        f"Pred    - Min: {all_preds.min():.4f}, Max: {all_preds.max():.4f}\n"
    )
    
    ax5.text(
        0.05, 0.95, metrics_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.suptitle('RFID 1D Localization - LSTM Model Evaluation', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plot_path = f"{output_dir}/evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {plot_path}")
    
    plt.show()


def evaluate_model_continuous(
    model_path: str,
    use_val: bool = False,
    sequence_length: int = 100,
    device: str = "cuda",
    eval_mode: Literal["final", "all"] = "final",
    save_plots: bool = True,
    output_dir: str = ".",
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
        save_plots: Whether to save visualization plots.
        output_dir: Directory to save plots.
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

    # Create visualizations
    if save_plots:
        print("\n=== GENERATING PLOTS ===")
        _create_visualizations(
            all_preds_np,
            all_trues_np,
            mse,
            rmse,
            mae,
            output_dir=output_dir,
        )

    return mse, rmse, mae


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continuous 1D location LSTM model with visualizations"
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    evaluate_model_continuous(
        model_path=args.model_path,
        use_val=args.use_val,
        sequence_length=args.sequence_length,
        device=args.device,
        eval_mode=args.eval_mode,  # type: ignore[arg-type]
        save_plots=not args.no_plots,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()