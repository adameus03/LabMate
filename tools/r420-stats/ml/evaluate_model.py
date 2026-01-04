import torch
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix


def evaluate_model_sequences(model_path, sequence_length=100, device='cuda', eval_mode='final'):
    """
    Evaluate trained model with sequences (compatible with training)
    
    Args:
        model_path: Path to model checkpoint
        sequence_length: Must match training sequence length
        device: 'cuda' or 'cpu'
        eval_mode: 'final' (only last timestep) or 'all' (all timesteps)
    """
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
    
    print(f"\nTotal validation samples: {len(all_inputs)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Evaluation mode: {eval_mode}")
    
    # Check ground truth distribution
    true_class_labels = np.argmax(all_outputs, axis=1)
    print(f"\nGround truth class distribution:")
    unique, counts = np.unique(true_class_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(true_class_labels):.2f}%)")
    
    # Load model
    print("\n=== MODEL LOADING ===")
    model = Model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")
    
    # Create sequences
    num_sequences = len(all_inputs) - sequence_length + 1
    print(f"\nNumber of sequences: {num_sequences}")
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        batch_size = 32  # Process 32 sequences at a time
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        print("\n=== GENERATING PREDICTIONS ===")
        for batch_idx in range(num_batches):
            start_seq = batch_idx * batch_size
            end_seq = min((batch_idx + 1) * batch_size, num_sequences)
            current_batch_size = end_seq - start_seq
            
            # Create batch of sequences
            batch_sequences_input = []
            batch_sequences_output = []
            
            for seq_idx in range(start_seq, end_seq):
                seq_input = all_inputs[seq_idx:seq_idx + sequence_length]
                seq_output = all_outputs[seq_idx:seq_idx + sequence_length]
                batch_sequences_input.append(seq_input)
                batch_sequences_output.append(seq_output)
            
            batch_inputs = torch.FloatTensor(np.array(batch_sequences_input)).to(device)
            batch_outputs = torch.FloatTensor(np.array(batch_sequences_output)).to(device)
            # batch_inputs shape: (batch_size, sequence_length, 208)
            # batch_outputs shape: (batch_size, sequence_length, 3)
            
            # Initialize states for entire batch
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
            
            # Process sequence timestep by timestep
            for t in range(sequence_length):
                c1, c2, c3, c4, c5, h1, h2, h3, h4, h5 = model(
                    batch_inputs[:, t, :],
                    (c1, c2, c3, c4, c5),
                    (h1, h2, h3, h4, h5)
                )
                
                # Collect predictions based on eval_mode
                if eval_mode == 'all' or (eval_mode == 'final' and t == sequence_length - 1):
                    predicted_classes = torch.argmax(h5, dim=1).cpu().numpy()
                    true_classes = torch.argmax(batch_outputs[:, t, :], dim=1).cpu().numpy()
                    
                    all_predictions.extend(predicted_classes)
                    all_true_labels.extend(true_classes)
            
            if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
                print(f"Processed {batch_idx + 1}/{num_batches} batches")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    print(f"\n=== PREDICTION ANALYSIS ===")
    print(f"Total predictions: {len(all_predictions)}")
    
    print(f"\nPredicted class distribution:")
    pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
    for cls, count in zip(pred_unique, pred_counts):
        print(f"  Class {cls}: {count} predictions ({100*count/len(all_predictions):.2f}%)")
    
    print(f"\nTrue class distribution:")
    true_unique, true_counts = np.unique(all_true_labels, return_counts=True)
    for cls, count in zip(true_unique, true_counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(all_true_labels):.2f}%)")
    
    # Check for model collapse
    if len(pred_unique) == 1:
        print("\n⚠️  WARNING: Model is predicting only ONE class!")
        print("This indicates a collapsed model.")
    
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
    
    return cm, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RFID model with sequences')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--sequence_length', type=int, default=100, 
                        help='Sequence length (must match training)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--eval_mode', type=str, default='final', choices=['final', 'all'],
                        help='Evaluate only final timestep or all timesteps')
    
    args = parser.parse_args()
    
    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    
    evaluate_model_sequences(
        args.model_path, 
        sequence_length=args.sequence_length,
        device=device,
        eval_mode=args.eval_mode
    )
