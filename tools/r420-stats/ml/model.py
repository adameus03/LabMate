import torch
import torch.nn as nn
from typing import Tuple

class LSTMCell(nn.Module):
  def __init__(self, input_dimensionality: int, state_dimensionality: int):
    super().__init__()
    self.input_dimensionality = input_dimensionality
    self.state_dimensionality = state_dimensionality

    self.perceptrons = [
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
    ]

  # returns [new_cell_state, new_hidden_state]
  def forward(self, input: torch.Tensor, cell_state: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    forget_gate_output = torch.sigmoid(self.perceptrons[0](input) + self.perceptrons[1](hidden_state))
    input_gate_output = torch.sigmoid(self.perceptrons[2](input) + self.perceptrons[3](hidden_state)) * torch.tanh(self.perceptrons[4](input) + self.perceptrons[5](hidden_state))
    new_cell_state = cell_state * forget_gate_output + input_gate_output
    output_gate_output = torch.tanh(new_cell_state) * torch.sigmoid(self.perceptrons[6](input) + self.perceptrons[7](hidden_state))
    return new_cell_state, output_gate_output
  
####################### 
#  
#
#
#####################
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, input: torch.Tensor, cell_state: torch.Tensor, hidden_state: torch.Tensor):
