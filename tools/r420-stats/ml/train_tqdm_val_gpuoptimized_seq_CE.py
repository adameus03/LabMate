import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Optional
import numpy as np
from tqdm import tqdm


class BatchedSequenceDataset(Dataset):
    """Dataset for sequential RFID data with batching support"""
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, sequence_length: int):
        """
        Args:
            inputs: Shape (num_samples, 208) - flattened RFID measurements
            outputs: Shape (num_samples, 3) - one-hot encoded location labels
            sequence_length: Length of sequences to create (L)
        """
        self.inputs = torch.FloatTensor(inputs)
        # Convert one-hot to class indices for CrossEntropyLoss
        self.outputs = torch.LongTensor(np.argmax(outputs, axis=1))
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.inputs) - self.sequence_length + 1
    
    def __getitem__(self, idx: int):
        """Get a chronological sequence"""
        seq_inputs = self.inputs[idx:idx + self.sequence_length]
        seq_outputs = self.outputs[idx:idx + self.sequence_length]
        return seq_inputs, seq_outputs


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy"""
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_on_sequences_optimized(
    model: nn.Module, 
    train_datasets: List[BatchedSequenceDataset],
    val_datasets: Optional[List[BatchedSequenceDataset]] = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    use_amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    class_weights: Optional[torch.Tensor] = None
):
    """
    Optimized training with CrossEntropyLoss, batching, mixed precision, and DataLoader
    
    Args:
        model: Your Model instance
        train_datasets: List of training BatchedSequenceDataset objects
        val_datasets: Optional list of validation BatchedSequenceDataset objects
        num_epochs: Number of training epochs
        batch_size: Number of sequences to process in parallel
        learning_rate: Learning rate for optimizer
        device: 'cuda' or 'cpu'
        use_amp: Use automatic mixed precision (faster on modern GPUs)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        class_weights: Optional tensor of shape (3,) for handling class imbalance
    """
    model = model.to(device)
    
    # Use CrossEntropyLoss instead of MSE
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Setup automatic mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == 'cuda' else None
    
    # Combine all training datasets
    combined_train = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Combine validation datasets if provided
    val_loader = None
    if val_datasets is not None:
        combined_val = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            combined_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    best_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='seq', unit_scale=batch_size)
        
        for batch_inputs, batch_outputs in pbar:
            # batch_inputs: (batch_size, sequence_length, 208)
            # batch_outputs: (batch_size, sequence_length) - class indices
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_outputs = batch_outputs.to(device, non_blocking=True)
            
            current_batch_size = batch_inputs.size(0)
            sequence_length = batch_inputs.size(1)
            
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
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision if enabled
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                sequence_loss = 0.0
                sequence_acc = 0.0
                
                # Process sequence step by step (all batch items in parallel)
                for t in range(sequence_length):
                    # Forward pass for entire batch
                    c1, c2, c3, c4, c5, h1, h2, h3, h4, h5 = model(
                        batch_inputs[:, t, :],  # (batch_size, 208)
                        (c1, c2, c3, c4, c5),
                        (h1, h2, h3, h4, h5)
                    )
                    
                    # h5 shape: (batch_size, 3) - raw logits
                    # batch_outputs[:, t] shape: (batch_size,) - class indices
                    step_loss = criterion(h5, batch_outputs[:, t])
                    sequence_loss += step_loss
                    
                    # Calculate accuracy for this timestep
                    step_acc = calculate_accuracy(h5, batch_outputs[:, t])
                    sequence_acc += step_acc
                
                # Average loss and accuracy over sequence
                sequence_loss = sequence_loss / sequence_length
                sequence_acc = sequence_acc / sequence_length
            
            # Backpropagation
            if scaler is not None:
                scaler.scale(sequence_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                sequence_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += sequence_loss.item()
            total_acc += sequence_acc
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{sequence_loss.item():.4f}',
                'acc': f'{sequence_acc:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'avg_acc': f'{total_acc/num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            num_val_batches = 0
            
            val_pbar = tqdm(val_loader, desc='Validating', unit='seq', unit_scale=batch_size)
            
            with torch.no_grad():
                for batch_inputs, batch_outputs in val_pbar:
                    batch_inputs = batch_inputs.to(device, non_blocking=True)
                    batch_outputs = batch_outputs.to(device, non_blocking=True)
                    
                    current_batch_size = batch_inputs.size(0)
                    sequence_length = batch_inputs.size(1)
                    
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
                    
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        sequence_loss = 0.0
                        sequence_acc = 0.0
                        
                        for t in range(sequence_length):
                            c1, c2, c3, c4, c5, h1, h2, h3, h4, h5 = model(
                                batch_inputs[:, t, :],
                                (c1, c2, c3, c4, c5),
                                (h1, h2, h3, h4, h5)
                            )
                            
                            step_loss = criterion(h5, batch_outputs[:, t])
                            sequence_loss += step_loss
                            
                            step_acc = calculate_accuracy(h5, batch_outputs[:, t])
                            sequence_acc += step_acc
                        
                        sequence_loss = sequence_loss / sequence_length
                        sequence_acc = sequence_acc / sequence_length
                    
                    total_val_loss += sequence_loss.item()
                    total_val_acc += sequence_acc
                    num_val_batches += 1
                    
                    val_pbar.set_postfix({
                        'loss': f'{sequence_loss.item():.4f}',
                        'acc': f'{sequence_acc:.4f}',
                        'avg_loss': f'{total_val_loss/num_val_batches:.4f}',
                        'avg_acc': f'{total_val_acc/num_val_batches:.4f}'
                    })
            
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_acc = total_val_acc / num_val_batches
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save best model based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_val_acc = avg_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'val_accuracy': best_val_acc,
                }, 'best_rfid_model.pth')
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, LR: {current_lr:.6f} - New best! ✓')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, LR: {current_lr:.6f}')
        else:
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_rfid_model.pth')
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, '
                      f'LR: {current_lr:.6f} - New best! ✓')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, '
                      f'LR: {current_lr:.6f}')
    
    return best_loss, best_val_acc


# Example usage
if __name__ == "__main__":
    from model import Model  # Import your model

    print("Loading training data")
    
    # Load your 3 training datasets
    dataset1_inputs = np.load('dataset1_inputs.npy')
    dataset1_outputs = np.load('dataset1_outputs.npy')
    
    dataset2_inputs = np.load('dataset2_inputs.npy')
    dataset2_outputs = np.load('dataset2_outputs.npy')
    
    dataset3_inputs = np.load('dataset3_inputs.npy')
    dataset3_outputs = np.load('dataset3_outputs.npy')
    
    # Load validation datasets
    val1_inputs = np.load('val1_inputs.npy')
    val1_outputs = np.load('val1_outputs.npy')
    
    val2_inputs = np.load('val2_inputs.npy')
    val2_outputs = np.load('val2_outputs.npy')
    
    val3_inputs = np.load('val3_inputs.npy')
    val3_outputs = np.load('val3_outputs.npy')
    
    # Optional: Calculate class weights if you have imbalanced data
    all_train_outputs = np.concatenate([dataset1_outputs, dataset2_outputs, dataset3_outputs])
    class_counts = all_train_outputs.sum(axis=0)
    class_weights = torch.FloatTensor(len(all_train_outputs) / (3 * class_counts))
    print(f"Class weights: {class_weights}")
    print(f"Class distribution: {class_counts}")
    
    # Create sequence datasets
    L = 100  # Sequence length
    train_datasets = [
        BatchedSequenceDataset(dataset1_inputs, dataset1_outputs, L),
        BatchedSequenceDataset(dataset2_inputs, dataset2_outputs, L),
        BatchedSequenceDataset(dataset3_inputs, dataset3_outputs, L)
    ]
    
    val_datasets = [
        BatchedSequenceDataset(val1_inputs, val1_outputs, L),
        BatchedSequenceDataset(val2_inputs, val2_outputs, L),
        BatchedSequenceDataset(val3_inputs, val3_outputs, L)
    ]
    
    # Initialize model
    print("Initializing model")
    model = Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train with optimized settings
    print(f"Starting optimized training on device: {device}")
    best_loss, best_val_acc = train_on_sequences_optimized(
        model, 
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        num_epochs=10000,
        batch_size=1024,  # Process 1024 sequences in parallel
        learning_rate=0.001,
        device=device,
        use_amp=True,  # Use mixed precision for speed
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        class_weights=class_weights  # Use class weights if imbalanced
    )
    
    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}, Best validation accuracy: {best_val_acc:.4f}")
    print("Best model saved to 'best_rfid_model.pth'")