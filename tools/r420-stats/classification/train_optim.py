from dnn import DenseNeuralNetwork
import re
import jax.numpy as jnp
import jax
import sys
from tqdm import tqdm
import numpy as np

oscillator_period = 120
tbptt_window_size = 120
nn = DenseNeuralNetwork([130, 182, 104, 71])

frequency_hops = []

# Locus mapping
locus_map = {
    '[klawiatura]': 0,
    '[zdobywcy]': 1,
    '[krzeslo]': 2,
    '[obudowa]': 3,
    '[szuflady]': 4,
    '[salewa]': 5,
    '[none]': -1
}

# Reverse mapping for display
locus_names = {
    0: 'klawiatura',
    1: 'zdobywcy',
    2: 'krzeslo',
    3: 'obudowa',
    4: 'szuflady',
    5: 'salewa',
    -1: 'none'
}

# Global arrays for neural network state
global_inputs = jnp.zeros(130)
global_outputs = jnp.zeros(71)
loaded_data = None

def load_frequency_hops(path):
    """Load frequency hopping table from a file."""
    freq_re = re.compile(r'Channel Index: (\d+), Frequency: (\d+) kHz')
    freq_table = {}
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Loading FH table", unit="lines"):
        line = line.strip()
        if not line.startswith('Parsed FH Table entry'):
            continue
        match = freq_re.search(line)
        if match:
            channel_idx = int(match.group(1))
            frequency = int(match.group(2))
            freq_table[channel_idx] = frequency
    
    if not freq_table:
        print("Warning: No frequency hop entries found in file!")
        return None
    
    max_idx = max(freq_table.keys())
    freq_list = [freq_table.get(i, 915000) for i in range(max_idx + 1)]
    return freq_list

def load_data(path):
    """Load and parse RFID measurement data from a log file."""
    if not frequency_hops:
        print("Error: Frequency hopping table not loaded!")
        exit(1)
    if len(frequency_hops) < 50:
        print("Error: Frequency hopping table seems incomplete!")
        exit(1)

    epc_re = re.compile(r'EPC: ([0-9A-F]+)')
    rssi_re = re.compile(r'16-bit peak rssi: (-?\d+)')
    phase_re = re.compile(r'RF Phase Angle: (\d+)')
    doppler_re = re.compile(r'RF Doppler Frequency: (-?\d+)')
    timestamp_re = re.compile(r'Last Seen Timestamp \(UTC\): (\d+) us')
    channel_re = re.compile(r'Channel Index: (\d+)')

    measurements = []
    current_locus = -1

    with open(path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Loading data", unit="lines"):
        line = line.strip()

        if line in locus_map:
            current_locus = locus_map[line]
            continue

        if not line.startswith('Measurement entry'):
            continue

        epc_match = epc_re.search(line)
        if not epc_match:
            continue
        epc = epc_match.group(1)

        if epc == '14B004730000000000000000':
            tag_index = -1
        elif epc.startswith('14B00473DEADBEEF'):
            tag_hex = epc[-8:]
            tag_num = int(tag_hex, 16)
            tag_index = tag_num % 12
        else:
            continue

        peak_rssi_16bit = int(rssi_re.search(line).group(1))
        phase_angle = int(phase_re.search(line).group(1))
        doppler_freq = int(doppler_re.search(line).group(1))
        last_seen_us = int(timestamp_re.search(line).group(1))
        channel_index = int(channel_re.search(line).group(1))

        rssi_scaled = peak_rssi_16bit / 32768.0
        phase_scaled = (phase_angle / 4096.0) * 2.0 - 1.0
        doppler_scaled = doppler_freq / 32768.0
        freq_khz = frequency_hops[channel_index] if channel_index < len(frequency_hops) else 915000
        freq_scaled = (freq_khz - 915000) / 13000.0
        sine_value = jnp.sin((2 * jnp.pi * last_seen_us / 1e6) / oscillator_period)

        measurements.append([
            tag_index,
            current_locus,
            rssi_scaled,
            phase_scaled,
            doppler_scaled,
            freq_scaled,
            float(sine_value)
        ])

    return jnp.array(measurements) if measurements else jnp.zeros((0, 7))

@jax.jit
def process_single_entry(global_inputs, global_outputs, entry, nn_params):
    """
    Process a single measurement entry (JIT-compiled for speed).
    
    Returns:
        tuple: (updated_global_inputs, updated_global_outputs, loss_contribution)
    """
    tag_index = jnp.int32(entry[0])
    locus = jnp.int32(entry[1])
    measurement_values = entry[2:]
    
    # Update global input array with measurement values using dynamic update
    # start_idx = (tag_index + 1) * 5, and we update 5 consecutive elements
    start_idx = (tag_index + 1) * 5
    global_inputs = jax.lax.dynamic_update_slice(
        global_inputs, 
        measurement_values, 
        (start_idx,)
    )
    
    # Update feedback loop (static indices - no problem)
    global_inputs = global_inputs.at[65:130].set(global_outputs[6:71])
    
    # Run neural network forward pass
    inputs_batch = global_inputs.reshape(1, -1)
    outputs_batch = nn.forward(nn_params, inputs_batch)
    global_outputs = outputs_batch.flatten()
    
    # Calculate loss - need to handle dynamic indexing for target vector
    predicted_locus_vector = global_outputs[:6]
    
    # Create one-hot encoded target using scatter instead of conditional indexing
    target_locus_vector = jnp.zeros(6)
    # Use a mask to set the appropriate index to 1.0
    mask = jnp.arange(6) == locus
    mask = jnp.where((locus >= 0) & (locus < 6), mask, False)
    target_locus_vector = jnp.where(mask, 1.0, 0.0)
    
    distance = jnp.linalg.norm(predicted_locus_vector - target_locus_vector)
    
    return global_inputs, global_outputs, distance

def loss_cb():
    """
    Calculate loss by processing a randomly chosen window of measurements.
    """
    global global_inputs, global_outputs, loaded_data
    
    if loaded_data is None or loaded_data.shape[0] == 0:
        return jnp.array(0.0)
    
    n = loaded_data.shape[0]
    if n < tbptt_window_size:
        window = loaded_data
    else:
        start = np.random.randint(0, n - tbptt_window_size + 1)
        window = loaded_data[start : start + tbptt_window_size]
    
    # Reset global state
    initial_inputs = jnp.zeros(130)
    initial_outputs = jnp.zeros(71)
    
    # Use jax.lax.scan for better performance - processes sequence efficiently
    def scan_fn(carry, entry):
        curr_inputs, curr_outputs, curr_loss = carry
        new_inputs, new_outputs, loss_contrib = process_single_entry(
            curr_inputs, curr_outputs, entry, nn.params
        )
        return (new_inputs, new_outputs, curr_loss + loss_contrib), None
    
    initial_carry = (initial_inputs, initial_outputs, jnp.array(0.0))
    final_carry, _ = jax.lax.scan(scan_fn, initial_carry, window)
    
    _, _, accumulated_loss = final_carry
    
    # Cannot print here as this function gets traced during JIT compilation
    return accumulated_loss

def show_data_summary(data):
    """Display first 10 entries for each locus."""
    print(f"Loaded {data.shape[0]} measurements total\n")
    
    for locus_idx in sorted(locus_names.keys()):
        locus_data = data[data[:, 1] == locus_idx]
        count = locus_data.shape[0]
        name = locus_names[locus_idx]
        
        print(f"=== Locus: [{name}] (index: {locus_idx}) ===")
        print(f"Total measurements: {count}")
        
        if count > 0:
            print(f"First {min(10, count)} entries:")
            print(locus_data[:10])
        else:
            print("No measurements for this locus")
        print()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python script.py <fh_table_file> <data_file> --show")
        print("  python script.py <fh_table_file> <data_file> --train <num_epochs>")
        sys.exit(1)
    
    fh_table_path = sys.argv[1]
    data_path = sys.argv[2]
    
    # Load frequency hopping table
    print(f"Loading frequency hopping table from {fh_table_path}...")
    loaded_fh_table = load_frequency_hops(fh_table_path)
    
    if loaded_fh_table is not None:
        frequency_hops = loaded_fh_table
        print(f"Done loading frequency hopping table. Loaded {len(frequency_hops)} channels.\n")
        print("Frequency Hopping Table:")
        for idx, freq in enumerate(frequency_hops[:10]):  # Show first 10
            print(f"  Channel {idx}: {freq} kHz")
        if len(frequency_hops) > 10:
            print(f"  ... and {len(frequency_hops) - 10} more channels")
        print()
    else:
        print("Failed to load frequency hopping table.")
        sys.exit(1)
    
    # Load training data
    print(f"Loading training data from {data_path}...")
    loaded_data = load_data(data_path)
    print("Done loading training data.\n")
    
    if len(sys.argv) >= 4:
        if sys.argv[3] == '--show':
            show_data_summary(loaded_data)
        elif sys.argv[3] == '--train' and len(sys.argv) >= 5:
            num_epochs = int(sys.argv[4])
            print(f"Training for {num_epochs} epochs...")
            print("Note: First epoch will be slower due to JIT compilation.\n")

            for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch"):
                # Train and capture the loss if possible
                # Note: nn.train() may not return loss, so this is just the training call
                nn.train(loss_callback=loss_cb, epochs=1, device='cpu')
                
                # Optionally compute and display loss every N epochs
                if epoch % 10 == 0:
                    # Call loss_cb directly to get current loss (outside of gradient computation)
                    current_loss = loss_cb()
                    tqdm.write(f"Epoch {epoch}: Loss = {float(current_loss):.4f}")

                # Save parameters periodically
                if epoch % 100 == 0 or epoch == num_epochs:
                    filename = f"nn_params_epoch_{epoch}.npz"
                    save_dict = {}
                    for i, layer_params in enumerate(nn.params):
                        if isinstance(layer_params, tuple):
                            save_dict[f'layer_{i}_weights'] = np.array(layer_params[0])
                            if len(layer_params) > 1:
                                save_dict[f'layer_{i}_biases'] = np.array(layer_params[1])
                        else:
                            save_dict[f'layer_{i}'] = np.array(layer_params)
                    np.savez(filename, **save_dict)
                    tqdm.write(f"Saved parameters to {filename}")

            print("\nTraining complete!")
        else:
            print("Invalid arguments. Use --show or --train <num_epochs>")
    else:
        print("Please specify --show or --train <num_epochs>")