import argparse
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from model import Model


class BatchedSequenceDataset(Dataset):
    """Dataset for sequential RFID data with continuous 1D targets."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, sequence_length: int):
        """
        Args:
            inputs: Shape (num_samples, 416) - flattened RFID measurements.
            targets: Shape (num_samples, 1) - scalar coordinates in [0, 1].
            sequence_length: Length of sequences to create (L).
        """
        assert inputs.shape[0] == targets.shape[0], "Inputs and targets length mismatch"
        self.inputs = torch.FloatTensor(inputs)
        # Store as 1D tensor for convenience: (num_samples,)
        self.targets = torch.FloatTensor(targets.squeeze(-1))
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.inputs) - self.sequence_length + 1

    def __getitem__(self, idx: int):
        """
        Returns:
            seq_inputs:  (L, 416)
            seq_targets: (L,)  continuous coordinates
        """
        seq_inputs = self.inputs[idx : idx + self.sequence_length]
        seq_targets = self.targets[idx : idx + self.sequence_length]
        return seq_inputs, seq_targets


def train_on_sequences_continuous(
    model: nn.Module,
    train_datasets: List[BatchedSequenceDataset],
    val_datasets: Optional[List[BatchedSequenceDataset]] = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    use_amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Optimized training loop for continuous 1D coordinate regression using MSE loss.
    """
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    combined_train = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = None
    if val_datasets is not None:
        combined_val = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            combined_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="seq",
            unit_scale=batch_size,
        )

        for batch_inputs, batch_targets in pbar:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            current_batch_size = batch_inputs.size(0)
            sequence_length = batch_inputs.size(1)

            c1 = torch.zeros(current_batch_size, 200, device=device)
            c2 = torch.zeros(current_batch_size, 96, device=device)
            c3 = torch.zeros(current_batch_size, 1, device=device)

            h1 = torch.zeros(current_batch_size, 200, device=device)
            h2 = torch.zeros(current_batch_size, 96, device=device)
            h3 = torch.zeros(current_batch_size, 1, device=device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                sequence_loss = 0.0

                for t in range(sequence_length):
                    c1, c2, c3, h1, h2, h3 = model(
                        batch_inputs[:, t, :],
                        (c1, c2, c3),
                        (h1, h2, h3),
                    )
                    # h3: (batch_size, 1)
                    step_pred = h3.squeeze(-1)  # (batch_size,)
                    step_target = batch_targets[:, t]  # (batch_size,)
                    step_loss = criterion(step_pred, step_target)
                    sequence_loss = sequence_loss + step_loss

                sequence_loss = sequence_loss / sequence_length

            if scaler is not None:
                scaler.scale(sequence_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                sequence_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += sequence_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{sequence_loss.item():.6f}",
                    "avg_loss": f"{total_loss / num_batches:.6f}",
                }
            )

        avg_loss = total_loss / num_batches

        # Validation
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0

            val_pbar = tqdm(
                val_loader,
                desc="Validating",
                unit="seq",
                unit_scale=batch_size,
            )

            with torch.no_grad():
                for batch_inputs, batch_targets in val_pbar:
                    batch_inputs = batch_inputs.to(device, non_blocking=True)
                    batch_targets = batch_targets.to(device, non_blocking=True)

                    current_batch_size = batch_inputs.size(0)
                    sequence_length = batch_inputs.size(1)

                    c1 = torch.zeros(current_batch_size, 200, device=device)
                    c2 = torch.zeros(current_batch_size, 96, device=device)
                    c3 = torch.zeros(current_batch_size, 1, device=device)

                    h1 = torch.zeros(current_batch_size, 200, device=device)
                    h2 = torch.zeros(current_batch_size, 96, device=device)
                    h3 = torch.zeros(current_batch_size, 1, device=device)

                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        sequence_loss = 0.0

                        for t in range(sequence_length):
                            c1, c2, c3, h1, h2, h3 = model(
                                batch_inputs[:, t, :],
                                (c1, c2, c3),
                                (h1, h2, h3),
                            )
                            step_pred = h3.squeeze(-1)
                            step_target = batch_targets[:, t]
                            step_loss = criterion(step_pred, step_target)
                            sequence_loss = sequence_loss + step_loss

                        sequence_loss = sequence_loss / sequence_length

                    total_val_loss += sequence_loss.item()
                    num_val_batches += 1

                    val_pbar.set_postfix(
                        {
                            "loss": f"{sequence_loss.item():.6f}",
                            "avg_loss": f"{total_val_loss / num_val_batches:.6f}",
                        }
                    )

            avg_val_loss = total_val_loss / num_val_batches
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                        "val_loss": avg_val_loss,
                    },
                    "continuous_best_rfid_model.pth",
                )
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"LR: {current_lr:.6f} - New best ✓"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )
        else:
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                    },
                    "continuous_best_rfid_model.pth",
                )
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_loss:.6f}, "
                    f"LR: {current_lr:.6f} - New best ✓"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )

    return best_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train continuous LSTM model for 1D location regression"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (number of sequences)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    use_amp = not args.no_amp

    print("Loading continuous datasets (.npy)")
    train_inputs = np.load("train_inputs.npy")
    train_targets = np.load("train_targets.npy")
    val_inputs = np.load("val_inputs.npy")
    val_targets = np.load("val_targets.npy")

    L_seq = args.sequence_length

    train_datasets = [
        BatchedSequenceDataset(train_inputs, train_targets, L_seq),
    ]
    val_datasets = [
        BatchedSequenceDataset(val_inputs, val_targets, L_seq),
    ]

    print("Initializing continuous model")
    model = Model()

    print(f"Starting training on device: {device}")
    best_loss = train_on_sequences_continuous(
        model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        use_amp=use_amp,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\nTraining finished. Best validation loss: {best_loss:.6f}")
    print("Best model saved to 'continuous_best_rfid_model.pth'")


if __name__ == "__main__":
    main()

