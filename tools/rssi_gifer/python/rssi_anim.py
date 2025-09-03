import sys
import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output.gif>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Config
    RSSI_MIN, RSSI_MAX = 150, 240   # adjust range if needed
    BINS = 40

    # --- load data ---
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if not files:
        print(f"No JSON files found in {input_dir}")
        sys.exit(1)

    datasets = []
    for fname in files:
        with open(fname) as f:
            data = json.load(f)
            vals = [m["rssi"] for m in data.get("measurements", [])
                    if RSSI_MIN <= m["rssi"] <= RSSI_MAX]
            print(f"Loaded {len(vals)} RSSI values from {fname}")
            datasets.append(vals)

    # --- prepare figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(RSSI_MIN, RSSI_MAX)
    ax.set_ylim(0, max(len(d) for d in datasets))  # constant y range
    ax.set_xlabel("RSSI")
    ax.set_ylabel("Count")

    # --- init / update functions ---
    def init():
        return []

    def update(frame):
        ax.cla()
        ax.set_xlim(RSSI_MIN, RSSI_MAX)
        ax.set_ylim(0, max(len(d) for d in datasets))
        ax.set_xlabel("RSSI")
        ax.set_ylabel("Count")
        ax.set_title(f"Frame {frame+1}: {os.path.basename(files[frame])}")
        ax.hist(datasets[frame], bins=BINS, color="steelblue", edgecolor="black")
        return ax.patches

    # --- animate ---
    ani = animation.FuncAnimation(fig, update, frames=len(datasets),
                                  init_func=init, blit=False, repeat=True)

    ani.save(output_file, writer="pillow", fps=5)
    print(f"Saved animation to {output_file}")

if __name__ == "__main__":
    main()
