import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import json
from pathlib import Path


class CustomLSTMCell(nn.Module):
    """Custom LSTM cell with explicit state management"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: input, forget, cell, output
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = states
        
        # Input gate
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        
        # Forget gate
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        
        # Cell gate
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_prev))
        
        # New cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        
        # New hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, (h_t, c_t)


class StackedBidirectionalLSTM(nn.Module):
    """Stacked Bidirectional LSTM for RFID RTLS localization"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.output_size = output_size
        
        # Create forward and backward LSTM cells for each layer
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        
        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i-1] * 2
            self.forward_cells.append(CustomLSTMCell(input_dim, hidden_size))
            self.backward_cells.append(CustomLSTMCell(input_dim, hidden_size))
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1] * 2, output_size)
        
    def init_states(self, batch_size: int, device: torch.device) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """Initialize hidden and cell states for all layers"""
        states = []
        for hidden_size in self.hidden_sizes:
            h_fwd = torch.zeros(batch_size, hidden_size, device=device)
            c_fwd = torch.zeros(batch_size, hidden_size, device=device)
            h_bwd = torch.zeros(batch_size, hidden_size, device=device)
            c_bwd = torch.zeros(batch_size, hidden_size, device=device)
            states.append(((h_fwd, c_fwd), (h_bwd, c_bwd)))
        return states
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            states: Optional initial states for each layer
        
        Returns:
            output: Predicted coordinates/probabilities (batch_size, output_size)
            final_states: Final states for each layer
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if states is None:
            states = self.init_states(batch_size, device)
        
        # Process each layer
        layer_input = x
        new_states = []
        
        for layer_idx in range(self.num_layers):
            fwd_cell = self.forward_cells[layer_idx]
            bwd_cell = self.backward_cells[layer_idx]
            
            fwd_state, bwd_state = states[layer_idx]
            
            # Forward pass
            fwd_outputs = []
            current_fwd_state = fwd_state
            for t in range(seq_len):
                h_t, current_fwd_state = fwd_cell(layer_input[:, t, :], current_fwd_state)
                fwd_outputs.append(h_t)
            fwd_outputs = torch.stack(fwd_outputs, dim=1)
            
            # Backward pass
            bwd_outputs = []
            current_bwd_state = bwd_state
            for t in range(seq_len - 1, -1, -1):
                h_t, current_bwd_state = bwd_cell(layer_input[:, t, :], current_bwd_state)
                bwd_outputs.append(h_t)
            bwd_outputs = torch.stack(bwd_outputs[::-1], dim=1)
            
            # Concatenate forward and backward outputs
            layer_output = torch.cat([fwd_outputs, bwd_outputs], dim=-1)
            
            # Apply dropout except for last layer
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            
            layer_input = layer_output
            new_states.append((current_fwd_state, current_bwd_state))
        
        # Use the last timestep output for prediction
        final_output = layer_output[:, -1, :]
        prediction = self.output_layer(final_output)
        
        return prediction, new_states


class RFIDDataset(Dataset):
    """Dataset for RFID RTLS measurements"""
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 10
    ):
        """
        Args:
            data: Array of shape (num_samples, features) where features include:
                  [scaled_phase, rssi, doppler, read_rate, channel_freq] 
                  for tracked tag + reference tags × num_antennas
            labels: Array of shape (num_samples, 3) for x, y, z coordinates
            seq_len: Sequence length for temporal windows
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len
        
        # Normalize features
        self.data_mean = self.data.mean(dim=0)
        self.data_std = self.data.std(dim=0) + 1e-8
        self.data = (self.data - self.data_mean) / self.data_std
        
    def __len__(self):
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_len]
        label = self.labels[idx + self.seq_len - 1]
        return sequence, label


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_path: str = 'rfid_lstm_model.pth'
):
    """Train the LSTM model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                predictions, _ = model(sequences)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)
            print(f'Model saved with validation loss: {val_loss:.4f}')
    
    return train_losses, val_losses


def inference(
    model: nn.Module,
    sequence: np.ndarray,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    states: Optional[List] = None
) -> Tuple[np.ndarray, List]:
    """
    Perform inference on a single sequence
    
    Args:
        model: Trained model
        sequence: Input sequence of shape (seq_len, features)
        data_mean: Mean used for normalization
        data_std: Std used for normalization
        device: Device to run inference on
        states: Optional previous states for streaming inference
    
    Returns:
        prediction: Predicted coordinates (3,)
        new_states: Updated states for next inference
    """
    model.eval()
    
    # Normalize input
    sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
    sequence = (sequence - data_mean) / data_std
    sequence = sequence.to(device)
    
    with torch.no_grad():
        prediction, new_states = model(sequence, states)
    
    return prediction.cpu().numpy()[0], new_states


# Example usage
if __name__ == "__main__":
    # Configuration
    NUM_ANTENNAS = 4
    NUM_REFERENCE_TAGS = 3
    FEATURES_PER_TAG = 5  # scaled_phase, rssi, doppler, read_rate, channel_freq
    INPUT_SIZE = (1 + NUM_REFERENCE_TAGS) * NUM_ANTENNAS * FEATURES_PER_TAG
    HIDDEN_SIZES = [128, 64, 32]
    SEQ_LEN = 10
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    
    print(f"Input size: {INPUT_SIZE} features")
    print(f"Hidden layers: {HIDDEN_SIZES}")
    print(f"Sequence length: {SEQ_LEN}")
    
    # Generate synthetic data for demonstration
    # In practice, replace this with your actual RFID data
    num_samples = 10000
    synthetic_data = np.random.randn(num_samples, INPUT_SIZE)
    synthetic_labels = np.random.randn(num_samples, 3) * 10  # x, y, z coordinates
    
    # Split data
    train_size = int(0.8 * num_samples)
    train_data = synthetic_data[:train_size]
    train_labels = synthetic_labels[:train_size]
    val_data = synthetic_data[train_size:]
    val_labels = synthetic_labels[train_size:]
    
    # Create datasets
    train_dataset = RFIDDataset(train_data, train_labels, seq_len=SEQ_LEN)
    val_dataset = RFIDDataset(val_data, val_labels, seq_len=SEQ_LEN)
    
    # Save normalization parameters
    norm_params = {
        'mean': train_dataset.data_mean.numpy().tolist(),
        'std': train_dataset.data_std.numpy().tolist()
    }
    with open('normalization_params.json', 'w') as f:
        json.dump(norm_params, f)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = StackedBidirectionalLSTM(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        output_size=3,
        dropout=0.2
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=0.001,
        device=device,
        save_path='rfid_lstm_model.pth'
    )
    
    # Example inference
    print("\n--- Inference Example ---")
    
    # Load normalization parameters
    with open('normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    data_mean = torch.FloatTensor(norm_params['mean'])
    data_std = torch.FloatTensor(norm_params['std'])
    
    # Load best model
    checkpoint = torch.load('rfid_lstm_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create a test sequence
    test_sequence = np.random.randn(SEQ_LEN, INPUT_SIZE)
    
    # Perform inference
    prediction, states = inference(
        model=model,
        sequence=test_sequence,
        data_mean=data_mean,
        data_std=data_std,
        device=device
    )
    
    print(f"Predicted coordinates: x={prediction[0]:.2f}, y={prediction[1]:.2f}, z={prediction[2]:.2f}")
    
    # Streaming inference example
    print("\n--- Streaming Inference Example ---")
    current_states = None
    for i in range(3):
        new_measurement = np.random.randn(SEQ_LEN, INPUT_SIZE)
        prediction, current_states = inference(
            model=model,
            sequence=new_measurement,
            data_mean=data_mean,
            data_std=data_std,
            device=device,
            states=current_states
        )
        print(f"Step {i+1} - Predicted: x={prediction[0]:.2f}, y={prediction[1]:.2f}, z={prediction[2]:.2f}")