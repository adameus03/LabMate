from dnn import DenseNeuralNetwork
import re
import jax.numpy as jnp
import sys
from tqdm import tqdm  # added tqdm for progress bars
import numpy as np  # needed for saving params

oscillator_period = 120
tbptt_window_size = 120
nn = DenseNeuralNetwork([130, 182, 104, 71])

# Predefined frequency hops in kHz (default values, will be updated from file)
# frequency_hops = [
#     902750, 903250, 903750, 904250, 904750, 905250, 905750, 906250,
#     906750, 907250, 907750, 908250, 908750, 909250, 909750, 910250,
#     910750, 911250, 911750, 912250, 912750, 913250, 913750, 914250,
#     914750, 915250, 915750, 916250, 916750, 917250, 917750, 918250,
#     918750, 919250, 919750, 920250, 920750, 921250, 921750, 922250,
#     922750, 923250, 923750, 924250, 924750, 925250, 925750, 926250,
#     926750, 927250, 927750
# ]
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
global_inputs = jnp.zeros(130)  # 130 input neurons
global_outputs = jnp.zeros(71)  # 71 output neurons
loaded_data = None  # Will store the data loaded from file

def load_frequency_hops(path):
    """
    Load frequency hopping table from a file.
    
    Args:
        path (str): Path to the frequency hopping table file.
    
    Returns:
        list: List of frequencies in kHz, indexed by channel index.
    """
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
    
    # Convert to list, sorted by channel index
    if not freq_table:
        print("Warning: No frequency hop entries found in file!")
        return None
    
    max_idx = max(freq_table.keys())
    freq_list = [freq_table.get(i, 915000) for i in range(max_idx + 1)]
    
    return freq_list

def load_data(path):
    """
    Load and parse RFID measurement data from a log file.
    
    Returns:
        jnp.ndarray: Array of shape (n_measurements, 7) where each row contains:
            - tag index (0-11 for reference tags, -1 for tracked tag)
            - index of current locus (0-5) or -1 if [none]
            - 16-bit peak rssi scaled from [-32768, 32767] to [-1, 1]
            - RF Phase Angle scaled from [0, 4096] to [-1, 1]
            - RF Doppler Frequency scaled from [-32768, 32767] to [-1, 1]
            - kHz frequency scaled from [902000, 928000] to [-1, 1]
            - sine oscillator value based on timestamp
    """
    # assert frequency_hops is loaded
    if not frequency_hops:
        print("Error: Frequency hopping table not loaded!")
        exit(1)
    if len(frequency_hops) < 50:
        print("Error: Frequency hopping table seems incomplete!")
        exit(1)

    # Precompile regexes for speed
    epc_re = re.compile(r'EPC: ([0-9A-F]+)')
    rssi_re = re.compile(r'16-bit peak rssi: (-?\d+)')
    phase_re = re.compile(r'RF Phase Angle: (\d+)')
    doppler_re = re.compile(r'RF Doppler Frequency: (-?\d+)')
    timestamp_re = re.compile(r'Last Seen Timestamp \(UTC\): (\d+) us')
    channel_re = re.compile(r'Channel Index: (\d+)')

    measurements = []
    current_locus = -1  # Start with [none]

    with open(path, 'r') as f:
        lines = f.readlines()

    # Use tqdm for progress bar
    for line in tqdm(lines, desc="Loading data", unit="lines"):
        line = line.strip()

        # Check for locus markers
        if line in locus_map:
            current_locus = locus_map[line]
            continue

        if not line.startswith('Measurement entry'):
            continue

        # Extract EPC
        epc_match = epc_re.search(line)
        if not epc_match:
            continue
        epc = epc_match.group(1)

        # Determine tag index
        if epc == '14B004730000000000000000':
            tag_index = -1  # Tracked tag
        elif epc.startswith('14B00473DEADBEEF'):
            tag_hex = epc[-8:]
            tag_num = int(tag_hex, 16)
            tag_index = tag_num % 12
        else:
            continue  # Unknown tag format

        # Extract measurement values
        peak_rssi_16bit = int(rssi_re.search(line).group(1))
        phase_angle = int(phase_re.search(line).group(1))
        doppler_freq = int(doppler_re.search(line).group(1))
        last_seen_us = int(timestamp_re.search(line).group(1))
        channel_index = int(channel_re.search(line).group(1))

        # Scale values to [-1, 1]
        rssi_scaled = peak_rssi_16bit / 32768.0
        phase_scaled = (phase_angle / 4096.0) * 2.0 - 1.0
        doppler_scaled = doppler_freq / 32768.0
        freq_khz = frequency_hops[channel_index] if channel_index < len(frequency_hops) else 915000
        freq_scaled = (freq_khz - 915000) / 13000.0
        sine_value = jnp.sin((2 * jnp.pi * last_seen_us / 1e6) / oscillator_period)

        # Append as Python list
        measurements.append([
            tag_index,
            current_locus,
            rssi_scaled,
            phase_scaled,
            doppler_scaled,
            freq_scaled,
            float(sine_value)
        ])

    # Convert once to JAX array at the end
    return jnp.array(measurements) if measurements else jnp.zeros((0, 7))

def loss_cb():
    """
    Calculate loss by processing a randomly chosen window of measurements 
    of length tbptt_window_size and comparing network outputs
    to expected locus values.
    """
    global global_inputs, global_outputs, loaded_data
    
    if loaded_data is None or loaded_data.shape[0] == 0:
        return jnp.array(0.0)
    
    n = loaded_data.shape[0]
    if n < tbptt_window_size:
        # If data is smaller than window, process all
        window = loaded_data
    else:
        # Choose a random contiguous window
        start = np.random.randint(0, n - tbptt_window_size + 1)
        window = loaded_data[start : start + tbptt_window_size]
    
    # Reset global state at the beginning
    global_inputs = jnp.zeros(130)
    global_outputs = jnp.zeros(71)
    
    accumulated_loss = jnp.array(0.0)
    
    # Use tqdm for progress bar over the window
    for entry in tqdm(window, desc="Processing window", unit="entries"):
        tag_index = int(entry[0])  # -1 for tracked tag, 0-11 for reference tags
        locus = int(entry[1])  # -1 for none, 0-5 for loci
        measurement_values = entry[2:]  # 5 values: rssi, phase, doppler, freq, sine
        
        # Update global input array with measurement values
        start_idx = (tag_index + 1) * 5
        end_idx = start_idx + 5
        global_inputs = global_inputs.at[start_idx:end_idx].set(measurement_values)
        
        # Update feedback loop: indices 65 to 129 (13*5+1 to 129)
        global_inputs = global_inputs.at[65:130].set(global_outputs[6:71])
        
        # Run neural network forward pass
        inputs_batch = global_inputs.reshape(1, -1)
        outputs_batch = nn.forward(nn.params, inputs_batch)
        
        # Update global outputs
        global_outputs = outputs_batch.flatten()
        
        # Calculate loss based on locus prediction
        predicted_locus_vector = global_outputs[:6]
        
        # Create target vector
        target_locus_vector = jnp.zeros(6)
        if 0 <= locus <= 5:
            target_locus_vector = target_locus_vector.at[locus].set(1.0)
        
        # Calculate Euclidean distance
        distance = jnp.linalg.norm(predicted_locus_vector - target_locus_vector)
        accumulated_loss += distance
    
    print(f"Accumulated loss for window: {accumulated_loss}")
    return accumulated_loss

def show_data_summary(data):
    """
    Display first 10 entries for each locus.
    
    Args:
        data (jnp.ndarray): Loaded measurement data.
    """
    print(f"Loaded {data.shape[0]} measurements total\n")
    
    # Group by locus and print first 10 for each
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
    
    # Load frequency hopping table first
    print(f"Loading frequency hopping table from {fh_table_path}...")
    loaded_fh_table = load_frequency_hops(fh_table_path)
    
    if loaded_fh_table is not None:
        frequency_hops = loaded_fh_table
        print(f"Done loading frequency hopping table. Loaded {len(frequency_hops)} channels.\n")
        print("Frequency Hopping Table:")
        for idx, freq in enumerate(frequency_hops):
            print(f"  Channel {idx}: {freq} kHz")
        print()
    else:
        print("Failed to load frequency hopping table. Using default values.\n")
    
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

            # Outer loop: epochs
            for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch"):
                # Train for 1 epoch at a time
                nn.train(loss_callback=loss_cb, epochs=1, device='cpu')

                # Save network parameters after each epoch
                if epoch % 100 == 0 or epoch == num_epochs:
                  filename = f"nn_params_epoch_{epoch}.npz"
                  save_dict = {}
                  for i, layer_params in enumerate(nn.params):
                      if isinstance(layer_params, tuple):
                          # Assume tuple is (weights, biases)
                          save_dict[f'layer_{i}_weights'] = np.array(layer_params[0])
                          if len(layer_params) > 1:
                              save_dict[f'layer_{i}_biases'] = np.array(layer_params[1])
                      else:
                          save_dict[f'layer_{i}'] = np.array(layer_params)
                  np.savez(filename, **save_dict)
                  tqdm.write(f"Saved network parameters to {filename}")

            print("Training complete!")
        else:
            print("Invalid arguments. Use --show or --train <num_epochs>")
    else:
        print("Please specify --show or --train <num_epochs>")