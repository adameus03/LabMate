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
    # Convert one-hot to class indices
    true_labels_tensor = torch.LongTensor(np.argmax(all_outputs, axis=1)).to(device)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(all_inputs)}")
    print(f"  Input shape: {all_inputs.shape}")
    
    # Check true label distribution
    true_labels_np = np.argmax(all_outputs, axis=1)
    unique, counts = np.unique(true_labels_np, return_counts=True)
    print(f"\nTrue label distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(true_labels_np):.1f}%)")
    
    # Load model
    print("\nLoading model...")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if 'val_accuracy' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.4f}")
    
    # Get predictions - process sample by sample to match training
    print("\nGenerating predictions...")
    all_predictions = []
    
    with torch.no_grad():
        batch_size = 1024
        num_batches = (len(inputs_tensor) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(inputs_tensor))
            
            batch_inputs = inputs_tensor[start_idx:end_idx]
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
            
            # Get class predictions from logits
            predicted_classes = torch.argmax(h5, dim=1).cpu().numpy()
            all_predictions.extend(predicted_classes)
            
            # Debug output for first batch
            if i == 0:
                print(f"\nSample predictions from first batch:")
                print(f"  Raw logits (first 5):")
                for j in range(min(5, len(h5))):
                    print(f"    Sample {j}: {h5[j].cpu().numpy()} -> Pred: {predicted_classes[j]}, True: {true_labels_tensor[j].item()}")
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_true_labels = true_labels_tensor.cpu().numpy()
    
    # Calculate overall accuracy
    correct = (all_predictions == all_true_labels).sum()
    total = len(all_true_labels)
    accuracy = correct / total
    
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({correct}/{total} correct)")
    print(f"{'='*60}")
    
    # Analyze predictions
    print(f"\nPredicted class distribution:")
    pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
    for cls, count in zip(pred_unique, pred_counts):
        print(f"  Class {cls}: {count} predictions ({100*count/len(all_predictions):.1f}%)")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, 2])
    
    # Print confusion matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print("              Predicted")
    print("           Class 0  Class 1  Class 2")
    print("         " + "-"*35)
    for i, row in enumerate(cm):
        row_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"True {i}  |  {row[0]:6d}   {row[1]:6d}   {row[2]:6d}  (acc: {row_acc:.3f})")
    print(f"{'='*60}")
    
    # Calculate per-class metrics
    print(f"\nPER-CLASS METRICS:")
    print(f"{'-'*60}")
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Class {i}:")
        print(f"  Precision: {precision:.4f}  (TP={tp}, FP={fp})")
        print(f"  Recall:    {recall:.4f}  (TP={tp}, FN={fn})")
        print(f"  F1-Score:  {f1:.4f}")
        print()
    
    # Identify most confused pairs
    print(f"MOST COMMON MISCLASSIFICATIONS:")
    print(f"{'-'*60}")
    misclass = []
    for i in range(3):
        for j in range(3):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], i, j))
    misclass.sort(reverse=True)
    for count, true_cls, pred_cls in misclass[:5]:
        pct = 100 * count / cm[true_cls, :].sum()
        print(f"  True {true_cls} → Predicted {pred_cls}: {count} times ({pct:.1f}% of class {true_cls})")
    
    return accuracy, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RFID model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    accuracy, cm = evaluate_model(args.model_path, device=device)
