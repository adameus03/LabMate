import torch
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix


def evaluate_model(model_path, device='cuda'):
    """Evaluate trained model and print confusion matrix"""
    from model import Model
    
    # Load validation datasets
    print("Loading validation data...")
    val1_inputs = np.load('val1_inputs.npy')
    val1_outputs = np.load('val1_outputs.npy')
    
    val2_inputs = np.load('val2_inputs.npy')
    val2_outputs = np.load('val2_outputs.npy')
    
    val3_inputs = np.load('val3_inputs.npy')
    val3_outputs = np.load('val3_outputs.npy')
    
    # Combine all validation data
    all_inputs = np.concatenate([val1_inputs, val2_inputs, val3_inputs], axis=0)
    all_outputs = np.concatenate([val1_outputs, val2_outputs, val3_outputs], axis=0)
    
    # Convert to tensors
    inputs_tensor = torch.FloatTensor(all_inputs).to(device)
    outputs_tensor = torch.FloatTensor(all_outputs).to(device)
    
    # Load model
    print("Loading model...")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    print("Generating predictions...")
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        batch_size = 1024
        num_batches = (len(inputs_tensor) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(inputs_tensor))
            
            batch_inputs = inputs_tensor[start_idx:end_idx]
            batch_outputs = outputs_tensor[start_idx:end_idx]
            current_batch_size = batch_inputs.size(0)
            
            # Initialize states
            c1 = torch.zeros(current_batch_size, 104, device=device)
            c2 = torch.zeros(current_batch_size, 52, device=device)
            c3 = torch.zeros(current_batch_size, 26, device=device)
            c4 = torch.zeros(current_batch_size, 10, device=device)
            c5 = torch.zeros(current_batch_size, 3, device=device)
            
            h1 = torch.zeros(current_batch_size, 104, device=device)
            h2 = torch.zeros(current_batch_size, 52, device=device)
            h3 = torch.zeros(current_batch_size, 26, device=device)
            h4 = torch.zeros(current_batch_size, 10, device=device)
            h5 = torch.zeros(current_batch_size, 3, device=device)
            
            # Forward pass
            c1, c2, c3, c4, c5, h1, h2, h3, h4, h5 = model(
                batch_inputs,
                (c1, c2, c3, c4, c5),
                (h1, h2, h3, h4, h5)
            )
            
            # Get class predictions
            predicted_classes = torch.argmax(h5, dim=1).cpu().numpy()
            true_classes = torch.argmax(batch_outputs, dim=1).cpu().numpy()
            
            all_predictions.extend(predicted_classes)
            all_true_labels.extend(true_classes)
    
    # Calculate confusion matrix
    cm = confusion_matrix(np.array(all_true_labels), np.array(all_predictions), labels=[0, 1, 2])
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate accuracy
    accuracy = np.trace(cm) / cm.sum()
    print(f"\nAccuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RFID model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    
    evaluate_model(args.model_path, device=device)