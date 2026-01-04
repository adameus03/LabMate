import torch
import torch.nn as nn
from typing import Tuple

class LSTMCell(nn.Module):
  def __init__(self, input_dimensionality: int, state_dimensionality: int):
    super().__init__()
    self.input_dimensionality = input_dimensionality
    self.state_dimensionality = state_dimensionality

    self.perceptrons = nn.ModuleList([
      # For forget gate
      nn.Linear(input_dimensionality, state_dimensionality),
      nn.Linear(state_dimensionality, state_dimensionality),
      # For input gate
      nn.Linear(input_dimensionality, state_dimensionality),
      nn.Linear(state_dimensionality, state_dimensionality),
      nn.Linear(input_dimensionality, state_dimensionality),
      nn.Linear(state_dimensionality, state_dimensionality),
      # For output gate
      nn.Linear(input_dimensionality, state_dimensionality),
      nn.Linear(state_dimensionality, state_dimensionality),
    ])

  # returns [new_cell_state, new_hidden_state]
  def forward(self, input: torch.Tensor, cell_state: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    forget_gate_output = torch.sigmoid(self.perceptrons[0](input) + self.perceptrons[1](hidden_state))
    input_gate_output = torch.sigmoid(self.perceptrons[2](input) + self.perceptrons[3](hidden_state)) * torch.tanh(self.perceptrons[4](input) + self.perceptrons[5](hidden_state))
    new_cell_state = cell_state * forget_gate_output + input_gate_output
    output_gate_output = torch.tanh(new_cell_state) * torch.sigmoid(self.perceptrons[6](input) + self.perceptrons[7](hidden_state))
    return new_cell_state, output_gate_output
  
#######################
#  rssi,phase,doppler,freq 
#  Input: 13 x 2 x 2 x 4 (num tags x num antennas x num measurements per tag per antenna x num features per measurement)
#  Output: 3 (num places -> piano, table segment 1, table segment 2)
#####################
class Model(nn.Module):
  def __init__(self, dropout_rate=0.3):
    super().__init__()
    self.stack = nn.ModuleList([
      LSTMCell(208, 104),
      LSTMCell(104, 52),
      LSTMCell(52, 26),
      LSTMCell(26,10),
      LSTMCell(10, 3)
    ])
    self.dropout = nn.Dropout(dropout_rate)
    
  def forward(self, input: torch.Tensor, cell_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], hidden_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c1,c2,c3,c4,c5 = cell_states
    h1,h2,h3,h4,h5 = hidden_states
    c1new,h1new = self.stack[0](input, c1, h1)
    h1new = self.dropout(h1new)
    c2new,h2new = self.stack[1](h1new, c2, h2)
    h2new = self.dropout(h2new)
    c3new,h3new = self.stack[2](h2new, c3, h3)
    h3new = self.dropout(h3new)
    c4new,h4new = self.stack[3](h3new, c4, h4)
    h4new = self.dropout(h4new)
    c5new,h5new = self.stack[4](h4new, c5, h5)
    return c1new,c2new,c3new,c4new,c5new,h1new,h2new,h3new,h4new,h5new

