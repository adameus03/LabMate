import argparse
import os
import sys
from typing import List

import numpy as np
import torch

from model import Model


def parse_csv_line(line: str, expected_dim: int = 416) -> np.ndarray:
    """
    Parse a single CSV line into a float32 numpy array of shape (expected_dim,).
    Returns None if parsing fails or dimensionality does not match.
    """
    line = line.strip()
    if not line:
        return None  # Skip empty lines

    parts = line.split(",")
    if len(parts) != expected_dim:
        sys.stderr.write(
            f"[WARN] Expected {expected_dim} columns, got {len(parts)}. Skipping line.\n"
        )
        sys.stderr.flush()
        return None

    try:
        values = np.array([float(x) for x in parts], dtype=np.float32)
    except ValueError:
        sys.stderr.write("[WARN] Failed to parse floats from line. Skipping.\n")
        sys.stderr.flush()
        return None

    return values


def run_stream_inference(
    model_path: str,
    pipe_path: str,
    sequence_length: int = 100,
    device: str = "cuda",
):
    """
    Read real-time CSV data from a named pipe and perform location inference.

    - Expects each line to be a CSV row with 416 float values (same as training).
    - Maintains a sliding window of the last `sequence_length` rows.
    - For each new row, once the window is full, runs the LSTM over the window
      and prints the final 1D coordinate prediction to stdout.
    """
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

    if not os.path.exists(pipe_path):
        raise FileNotFoundError(
            f"Named pipe '{pipe_path}' does not exist. "
            f"Create it with: mkfifo {pipe_path}"
        )

    if not stat_is_fifo(pipe_path):
        sys.stderr.write(
            f"[WARN] '{pipe_path}' is not a FIFO (named pipe). "
            f"Continuing anyway, but this script is intended for pipes.\n"
        )
        sys.stderr.flush()

    print(f"Loading model from {model_path} on device {device}")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Model loaded. Epoch={checkpoint.get('epoch', 'unknown')}, "
        f"loss={checkpoint.get('loss', 'unknown')}"
    )
    print(f"Waiting for data on pipe: {pipe_path}")
    sys.stdout.flush()

    buffer: List[np.ndarray] = []

    # Open the pipe for reading text lines
    with open(pipe_path, "r") as f:
        for line in f:
            sample = parse_csv_line(line)
            if sample is None:
                continue

            buffer.append(sample)
            if len(buffer) < sequence_length:
                continue  # Not enough history yet

            # Keep only the last `sequence_length` samples
            if len(buffer) > sequence_length:
                buffer = buffer[-sequence_length:]

            # Build (1, L, 416) tensor
            window = np.stack(buffer, axis=0).astype(np.float32)
            window_t = torch.FloatTensor(window[None, ...]).to(device)  # (1, L, 416)

            # Initialize hidden/cell states for a fresh sequence
            batch_size = 1
            c1 = torch.zeros(batch_size, 200, device=device)
            c2 = torch.zeros(batch_size, 96, device=device)
            c3 = torch.zeros(batch_size, 1, device=device)

            h1 = torch.zeros(batch_size, 200, device=device)
            h2 = torch.zeros(batch_size, 96, device=device)
            h3 = torch.zeros(batch_size, 1, device=device)

            with torch.no_grad():
                for t in range(sequence_length):
                    c1, c2, c3, h1, h2, h3 = model(
                        window_t[:, t, :],
                        (c1, c2, c3),
                        (h1, h2, h3),
                    )

                # h3: (1, 1) -> scalar
                coord = float(h3.item())

            # Print prediction (one per line) and flush so consumers see it immediately
            print(coord)
            sys.stdout.flush()


def stat_is_fifo(path: str) -> bool:
    """Return True if the path is a FIFO (named pipe)."""
    import stat
    st = os.stat(path)
    return stat.S_ISFIFO(st.st_mode)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time location inference from CSV data via named pipe"
    )
    parser.add_argument("model_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument(
        "pipe_path",
        type=str,
        help="Path to named pipe providing CSV rows (416 floats per line)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Sequence length (must match training for best results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )

    args = parser.parse_args()

    try:
        run_stream_inference(
            model_path=args.model_path,
            pipe_path=args.pipe_path,
            sequence_length=args.sequence_length,
            device=args.device,
        )
    except KeyboardInterrupt:
        sys.stderr.write("\n[INFO] Interrupted by user. Exiting.\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()

