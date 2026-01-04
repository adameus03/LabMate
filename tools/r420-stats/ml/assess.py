import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_path, device='cuda'):
    """
    Evaluate trained model and generate confusion matrix
    
    Args:
        model_path: Path to saved model checkpoint
        device: 'cuda' or 'cpu'
    """
    # Import model
    from model import Model
    
    # Load validation datasets
    print("Loading validation datasets...")
    val1_inputs = np.load('val1_inputs.npy')
    val1_outputs = np.load('val1_outputs.npy')
    
    val2_inputs = np.load('val2_inputs.npy')
    val2_outputs = np.load('val2_outputs.npy')
    
    val3_inputs = np.load('val3_inputs.npy')
    val3_outputs = np.load('val3_outputs.npy')
    
    # Combine all validation data
    all_inputs = np.concatenate([val1_inputs, val2_inputs, val3_inputs], axis=0)
    all_outputs = np.concatenate([val1_outputs, val2_outputs, val3_outputs], axis=0)
    
    print(f"Total validation samples: {len(all_inputs)}")
    
    # Convert to tensors
    inputs_tensor = torch.FloatTensor(all_inputs).to(device)
    outputs_tensor = torch.FloatTensor(all_outputs).to(device)
    
    # Initialize and load model
    print(f"Loading model from {model_path}...")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    
    # Get predictions
    print("Generating predictions...")
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
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
            
            # Convert predictions to class labels (argmax)
            predicted_classes = torch.argmax(h5, dim=1).cpu().numpy()
            true_classes = torch.argmax(batch_outputs, dim=1).cpu().numpy()
            
            all_predictions.extend(predicted_classes)
            all_true_labels.extend(true_classes)
            
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Processed batch {i+1}/{num_batches}")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, 2])
    
    # Print confusion matrix
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print("\nRaw counts:")
    print(cm)
    
    # Print with labels
    print("\n         Predicted")
    print("         Class 0  Class 1  Class 2")
    print("       +--------+--------+--------+")
    for i in range(3):
        row_label = f"Class {i}"
        print(f"{row_label:6} |{cm[i,0]:7} |{cm[i,1]:7} |{cm[i,2]:7} |")
        print("       +--------+--------+--------+")
    
    # Calculate and print metrics
    print("\n" + "="*50)
    print("METRICS PER CLASS")
    print("="*50)
    
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass {i}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix plot...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - RFID Location Classification')
    plt.tight_layout()
    
    # Save plot
    output_filename = 'confusion_matrix.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to '{output_filename}'")
    
    return cm, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RFID classification model')
    parser.add_argument('model_path', type=str, 
                       help='Path to trained model checkpoint (e.g., classifier.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run evaluation on (default: cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("Warning: CUDA not available, using CPU instead")
    
    print(f"Using device: {device}")
    
    # Run evaluation
    try:
        cm, accuracy = evaluate_model(args.model_path, device=device)
        print("\nEvaluation completed successfully!")
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise