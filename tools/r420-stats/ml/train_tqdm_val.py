import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

# Your existing model code here (LSTMCell and Model classes)

class SequenceDataset:
    """Dataset for sequential RFID data"""
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, sequence_length: int):
        """
        Args:
            inputs: Shape (num_samples, 208) - flattened RFID measurements
            outputs: Shape (num_samples, 3) - location probabilities/labels
            sequence_length: Length of sequences to create (L)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)
        self.sequence_length = sequence_length
        self.num_sequences = len(inputs) - sequence_length + 1
        
    def __len__(self):
        return self.num_sequences
    
    def get_sequence(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a chronological sequence"""
        seq_inputs = self.inputs[idx:idx + self.sequence_length]
        seq_outputs = self.outputs[idx:idx + self.sequence_length]
        return seq_inputs, seq_outputs


def train_on_sequences(model: nn.Module, 
                       datasets: List[SequenceDataset],
                       val_datasets: Optional[List[SequenceDataset]] = None,
                       num_epochs: int = 100,
                       sequence_length: int = 10,
                       learning_rate: float = 0.001,
                       device: str = 'cuda'):
    """
    Train the model on sequential data from multiple datasets
    
    Args:
        model: Your Model instance
        datasets: List of training SequenceDataset objects
        val_datasets: Optional list of validation SequenceDataset objects
        num_epochs: Number of training epochs
        sequence_length: Length L of sequences to process
        learning_rate: Learning rate for optimizer
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    criterion = nn.MSELoss()  # or nn.CrossEntropyLoss() if outputs are class labels
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_loss = float('inf')
    
    # Calculate total sequences across all datasets
    total_sequences = sum(len(dataset) for dataset in datasets)
    if val_datasets is not None:
        total_val_sequences = sum(len(dataset) for dataset in val_datasets)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_sequences = 0
        
        # Create progress bar for the epoch
        pbar = tqdm(total=total_sequences, 
                   desc=f'Epoch {epoch + 1}/{num_epochs}',
                   unit='seq')
        
        # Process each dataset
        for dataset_idx, dataset in enumerate(datasets):
            # Process all sequences in this dataset
            for seq_idx in range(len(dataset)):
                seq_inputs, seq_outputs = dataset.get_sequence(seq_idx)
                seq_inputs = seq_inputs.to(device)
                seq_outputs = seq_outputs.to(device)
                
                # Initialize states to zero for each sequence
                c1 = torch.zeros(104).to(device)
                c2 = torch.zeros(52).to(device)
                c3 = torch.zeros(26).to(device)
                c4 = torch.zeros(10).to(device)
                c5 = torch.zeros(3).to(device)
                
                h1 = torch.zeros(104).to(device)
                h2 = torch.zeros(52).to(device)
                h3 = torch.zeros(26).to(device)
                h4 = torch.zeros(10).to(device)
                h5 = torch.zeros(3).to(device)
                
                cell_states = (c1, c2, c3, c4, c5)
                hidden_states = (h1, h2, h3, h4, h5)
                
                optimizer.zero_grad()
                sequence_loss = 0.0
                
                # Process sequence step by step
                for t in range(sequence_length):
                    # Forward pass
                    c1, c2, c3, c4, c5, h1, h2, h3, h4, h5 = model(
                        seq_inputs[t], 
                        cell_states, 
                        hidden_states
                    )
                    
                    # Update states for next timestep
                    cell_states = (c1, c2, c3, c4, c5)
                    hidden_states = (h1, h2, h3, h4, h5)
                    
                    # Calculate loss for this timestep
                    # h5 is the output (3-dimensional)
                    step_loss = criterion(h5, seq_outputs[t])
                    sequence_loss += step_loss
                
                # Average loss over sequence
                sequence_loss = sequence_loss / sequence_length
                
                # Backpropagate through time
                sequence_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += sequence_loss.item()
                num_sequences += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{sequence_loss.item():.4f}',
                    'dataset': dataset_idx + 1,
                    'avg_loss': f'{total_loss/num_sequences:.4f}'
                })
                pbar.update(1)
        
        pbar.close()
        
        avg_loss = total_loss / num_sequences
        
        # Validation phase
        if val_datasets is not None:
            model.eval()
            total_val_loss = 0.0
            num_val_sequences = 0
            
            val_pbar = tqdm(total=total_val_sequences,
                           desc=f'Validating',
                           unit='seq')
            
            with torch.no_grad():
                for dataset_idx, dataset in enumerate(val_datasets):
                    for seq_idx in range(len(dataset)):
                        seq_inputs, seq_outputs = dataset.get_sequence(seq_idx)
                        seq_inputs = seq_inputs.to(device)
                        seq_outputs = seq_outputs.to(device)
                        
                        # Initialize states to zero for each sequence
                        c1 = torch.zeros(104).to(device)
                        c2 = torch.zeros(52).to(device)
                        c3 = torch.zeros(26).to(device)
                        c4 = torch.zeros(10).to(device)
                        c5 = torch.zeros(3).to(device)
                        
                        h1 = torch.zeros(104).to(device)
                        h2 = torch.zeros(52).to(device)
                        h3 = torch.zeros(26).to(device)
                        h4 = torch.zeros(10).to(device)
                        h5 = torch.zeros(3).to(device)
                        
                        cell_states = (c1, c2, c3, c4, c5)
                        hidden_states = (h1, h2, h3, h4, h5)
                        
                        sequence_loss = 0.0
                        
                        # Process sequence step by step
                        for t in range(sequence_length):
                            c1, c2, c3, c4, c5, h1, h2, h3, h4, h5 = model(
                                seq_inputs[t], 
                                cell_states, 
                                hidden_states
                            )
                            
                            cell_states = (c1, c2, c3, c4, c5)
                            hidden_states = (h1, h2, h3, h4, h5)
                            
                            step_loss = criterion(h5, seq_outputs[t])
                            sequence_loss += step_loss
                        
                        sequence_loss = sequence_loss / sequence_length
                        total_val_loss += sequence_loss.item()
                        num_val_sequences += 1
                        
                        val_pbar.set_postfix({
                            'loss': f'{sequence_loss.item():.4f}',
                            'dataset': dataset_idx + 1,
                            'avg_loss': f'{total_val_loss/num_val_sequences:.4f}'
                        })
                        val_pbar.update(1)
            
            val_pbar.close()
            avg_val_loss = total_val_loss / num_val_sequences
            scheduler.step(avg_val_loss)
            
            # Save best model based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_rfid_model.pth')
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f} - New best! ✓')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        else:
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_rfid_model.pth')
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f} - New best! ✓')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return best_loss


# Example usage
if __name__ == "__main__":
    from model import Model  # Import your model

    print("Loading training data")
    
    # Load your 3 training datasets
    dataset1_inputs = np.load('dataset1_inputs.npy')  # Piano data
    dataset1_outputs = np.load('dataset1_outputs.npy')
    
    dataset2_inputs = np.load('dataset2_inputs.npy')  # Table segment 1
    dataset2_outputs = np.load('dataset2_outputs.npy')
    
    dataset3_inputs = np.load('dataset3_inputs.npy')  # Table segment 2
    dataset3_outputs = np.load('dataset3_outputs.npy')
    
    # Load validation datasets (if available)
    val1_inputs = np.load('val1_inputs.npy')
    val1_outputs = np.load('val1_outputs.npy')
    
    val2_inputs = np.load('val2_inputs.npy')
    val2_outputs = np.load('val2_outputs.npy')
    
    val3_inputs = np.load('val3_inputs.npy')
    val3_outputs = np.load('val3_outputs.npy')
    
    # Create sequence datasets
    L = 100  # Sequence length
    train_datasets = [
        SequenceDataset(dataset1_inputs, dataset1_outputs, L),
        SequenceDataset(dataset2_inputs, dataset2_outputs, L),
        SequenceDataset(dataset3_inputs, dataset3_outputs, L)
    ]
    
    val_datasets = [
        SequenceDataset(val1_inputs, val1_outputs, L),
        SequenceDataset(val2_inputs, val2_outputs, L),
        SequenceDataset(val3_inputs, val3_outputs, L)
    ]
    
    # Initialize model
    print("Initializing model")
    model = Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    print(f"Starting to train on sequences using device: {device}")
    best_loss = train_on_sequences(
        model, 
        datasets=train_datasets,
        val_datasets=val_datasets,
        num_epochs=100, 
        sequence_length=L,
        learning_rate=0.001,
        device=device
    )
    
    print(f"\nTraining completed! Best loss: {best_loss:.6f}")