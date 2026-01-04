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
    
    # DEBUG: Check ground truth labels
    print("\n=== GROUND TRUTH LABELS DEBUG ===")
    print(f"all_outputs shape: {all_outputs.shape}")
    print(f"all_outputs dtype: {all_outputs.dtype}")
    print(f"Sample of all_outputs (first 5 rows):\n{all_outputs[:5]}")
    
    # Check if outputs are one-hot encoded
    if all_outputs.shape[-1] == 3:
        true_class_labels = np.argmax(all_outputs, axis=1)
        print(f"\nTrue class distribution:")
        unique, counts = np.unique(true_class_labels, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({100*count/len(true_class_labels):.2f}%)")
    else:
        print(f"WARNING: Expected one-hot encoded outputs with shape (N, 3), got {all_outputs.shape}")
    
    # Convert to tensors
    inputs_tensor = torch.FloatTensor(all_inputs).to(device)
    outputs_tensor = torch.FloatTensor(all_outputs).to(device)
    
    # Load model
    print("\n=== MODEL LOADING ===")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Get predictions
    print("\n=== GENERATING PREDICTIONS ===")
    all_predictions = []
    all_true_labels = []
    all_raw_outputs = []  # Store raw h5 outputs for debugging
    
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
            
            # DEBUG: Print raw outputs for first batch only
            if i == 0:
                print(f"\n=== RAW MODEL OUTPUTS (Batch 0) ===")
                print(f"h5 shape: {h5.shape}")
                print(f"h5 dtype: {h5.dtype}")
                print(f"Sample h5 values (first 10 samples):\n{h5[:10].cpu().numpy()}")
                print(f"\nMin values per class: {h5.min(dim=0).values.cpu().numpy()}")
                print(f"Max values per class: {h5.max(dim=0).values.cpu().numpy()}")
                print(f"Mean values per class: {h5.mean(dim=0).cpu().numpy()}")
                print(f"Std values per class: {h5.std(dim=0).cpu().numpy()}")
            
            # Get class predictions
            predicted_classes = torch.argmax(h5, dim=1).cpu().numpy()
            true_classes = torch.argmax(batch_outputs, dim=1).cpu().numpy()
            
            # Store raw outputs from first batch for analysis
            if i == 0:
                all_raw_outputs = h5.cpu().numpy()
            
            all_predictions.extend(predicted_classes)
            all_true_labels.extend(true_classes)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # DEBUG: Analyze predictions
    print("\n=== PREDICTION ANALYSIS ===")
    print(f"Total samples: {len(all_predictions)}")
    print(f"\nPredicted class distribution:")
    pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
    for cls, count in zip(pred_unique, pred_counts):
        print(f"  Class {cls}: {count} predictions ({100*count/len(all_predictions):.2f}%)")
    
    print(f"\nTrue class distribution:")
    true_unique, true_counts = np.unique(all_true_labels, return_counts=True)
    for cls, count in zip(true_unique, true_counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(all_true_labels):.2f}%)")
    
    # Check if model is predicting only one class
    if len(pred_unique) == 1:
        print("\n⚠️  WARNING: Model is predicting only ONE class!")
        print("This indicates a collapsed model. Possible causes:")
        print("  1. Model hasn't been trained properly")
        print("  2. Learning rate too high causing gradient explosion")
        print("  3. Class imbalance in training data")
        print("  4. Wrong loss function or improper normalization")
    
    # DEBUG: Raw output statistics (from first batch)
    if len(all_raw_outputs) > 0:
        print("\n=== RAW OUTPUT STATISTICS ===")
        print("Checking if outputs are saturated or collapsed:")
        for cls in range(3):
            cls_values = all_raw_outputs[:, cls]
            print(f"Class {cls}: min={cls_values.min():.6f}, max={cls_values.max():.6f}, "
                  f"mean={cls_values.mean():.6f}, std={cls_values.std():.6f}")
        
        # Check if one class dominates
        argmax_counts = np.bincount(np.argmax(all_raw_outputs, axis=1), minlength=3)
        print(f"\nArgmax distribution in first batch: {argmax_counts}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, 2])
    
    # Print confusion matrix
    print("\n=== CONFUSION MATRIX ===")
    print("         Predicted")
    print("       0    1    2")
    print("     ---------------")
    for i, row in enumerate(cm):
        print(f"  {i} | {row[0]:4d} {row[1]:4d} {row[2]:4d}")
    
    # Calculate per-class metrics
    print("\n=== PER-CLASS METRICS ===")
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Class {i}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # Calculate accuracy
    accuracy = np.trace(cm) / cm.sum()
    print(f"\n=== OVERALL ACCURACY ===")
    print(f"Accuracy: {accuracy:.4f} ({np.trace(cm)}/{cm.sum()} correct)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RFID model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    
    evaluate_model(args.model_path, device=device)