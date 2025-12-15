from dnn import DenseNeuralNetwork
import numpy as np
import sys

def load_network_params(npz_path):
    """
    Load neural network parameters from an .npz file.
    
    Args:
        npz_path (str): Path to the .npz file containing saved parameters.
    
    Returns:
        list: List of tuples (weights, biases) for each layer.
    """
    print(f"Loading parameters from {npz_path}...")
    
    # Load the npz file
    data = np.load(npz_path)
    
    # Print all keys to see what was saved
    print(f"Keys in file: {list(data.keys())}")
    
    # Reconstruct params as list of tuples
    params = []
    
    # Determine number of layers
    layer_indices = set()
    for key in data.keys():
        if key.startswith('layer_'):
            parts = key.split('_')
            layer_idx = int(parts[1])
            layer_indices.add(layer_idx)
    
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    print(f"Found {num_layers} layers")
    
    # Reconstruct each layer's parameters
    for i in range(num_layers):
        # Try both naming conventions
        if f'layer_{i}_weights' in data:
            weights = data[f'layer_{i}_weights']
            biases = data[f'layer_{i}_biases'] if f'layer_{i}_biases' in data else None
        elif f'layer_{i}_param_0' in data:
            weights = data[f'layer_{i}_param_0']
            biases = data[f'layer_{i}_param_1'] if f'layer_{i}_param_1' in data else None
        else:
            print(f"Warning: Could not find parameters for layer {i}")
            continue
        
        if biases is not None:
            params.append((weights, biases))
            print(f"  Layer {i}: weights shape {weights.shape}, biases shape {biases.shape}")
        else:
            params.append((weights,))
            print(f"  Layer {i}: weights shape {weights.shape}, no biases")
    
    return params


def create_network_with_params(layer_sizes, params):
    """
    Create a DenseNeuralNetwork and load parameters into it.
    
    Args:
        layer_sizes (list): List of layer sizes (e.g., [130, 182, 104, 71])
        params (list): List of parameter tuples loaded from file.
    
    Returns:
        DenseNeuralNetwork: Network with loaded parameters.
    """
    print(f"\nCreating network with architecture: {layer_sizes}")
    nn = DenseNeuralNetwork(layer_sizes)
    
    print(f"Loading {len(params)} parameter sets into network...")
    nn.params = params
    
    print("Network loaded successfully!")
    return nn


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python load_network.py <params_file.npz>")
        print("\nExample:")
        print("  python load_network.py ./nn_params_epoch_1.npz")
        sys.exit(1)
    
    params_path = sys.argv[1]
    
    # Load parameters
    loaded_params = load_network_params(params_path)
    
    # Create network with the same architecture as in train.py
    # You may need to adjust these layer sizes if your architecture is different
    layer_sizes = [130, 182, 104, 71]
    
    nn = create_network_with_params(layer_sizes, loaded_params)
    
    print("\n" + "="*60)
    print("Network ready for inference!")
    print("="*60)
    print(f"\nArchitecture: {layer_sizes}")
    print(f"Number of layers: {len(loaded_params)}")
    print(f"\nYou can now use 'nn' for inference:")
    print("  Example: outputs = nn.forward(nn.params, inputs)")
    
    # Optional: Save the network object for later use
    print("\nTo use this network in another script, you can:")
    print("  1. Import this script: from load_network import load_network_params, create_network_with_params")
    print("  2. Load params: params = load_network_params('path/to/file.npz')")
    print("  3. Create network: nn = create_network_with_params([130, 182, 104, 71], params)")