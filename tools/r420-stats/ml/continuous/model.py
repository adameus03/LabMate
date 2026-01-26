import torch
import torch.nn as nn
from typing import Tuple


class LSTMCell(nn.Module):
    """
    Custom LSTM cell identical to the classification version, reused here
    for a 1D continuous output model.
    """

    def __init__(self, input_dimensionality: int, state_dimensionality: int):
        super().__init__()
        self.input_dimensionality = input_dimensionality
        self.state_dimensionality = state_dimensionality

        self.perceptrons = nn.ModuleList(
            [
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
        )

    # returns [new_cell_state, new_hidden_state]
    def forward(
        self,
        input: torch.Tensor,
        cell_state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        forget_gate_output = torch.sigmoid(
            self.perceptrons[0](input) + self.perceptrons[1](hidden_state)
        )
        input_gate_output = torch.sigmoid(
            self.perceptrons[2](input) + self.perceptrons[3](hidden_state)
        ) * torch.tanh(
            self.perceptrons[4](input) + self.perceptrons[5](hidden_state)
        )
        new_cell_state = cell_state * forget_gate_output + input_gate_output
        output_gate_output = torch.tanh(new_cell_state) * torch.sigmoid(
            self.perceptrons[6](input) + self.perceptrons[7](hidden_state)
        )
        return new_cell_state, output_gate_output


#######################
#  rssi,phase,doppler,freq
#  Input: 13 x 4 x 2 x 4 (num tags x num antennas x num measurements per tag
#         per antenna x num features per measurement) -> 416-dim flattened
#  Output: 1 (continuous 1D location coordinate in [0, 1])
#######################
class Model(nn.Module):
    """
    LSTM stack with a 1D continuous output corresponding to a 1D location
    coordinate in [0, 1]. The training code is responsible for applying
    the appropriate regression loss on this scalar output.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.stack = nn.ModuleList(
            [
                LSTMCell(416, 200),
                LSTMCell(200, 96),
                LSTMCell(96, 1),
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        input: torch.Tensor,
        cell_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        hidden_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input: (batch_size, 416)
            cell_states: tuple of (c1, c2, c3)
            hidden_states: tuple of (h1, h2, h3)

        Returns:
            Updated (c1, c2, c3, h1, h2, h3) where h3 is the 1D location output.
        """
        c1, c2, c3 = cell_states
        h1, h2, h3 = hidden_states

        c1_new, h1_new = self.stack[0](input, c1, h1)
        h1_new = self.dropout(h1_new)

        c2_new, h2_new = self.stack[1](h1_new, c2, h2)
        h2_new = self.dropout(h2_new)

        c3_new, h3_new = self.stack[2](h2_new, c3, h3)
        # h3_new is of shape (<batchsize>, 1) and is used as the continuous coordinate
        return c1_new, c2_new, c3_new, h1_new, h2_new, h3_new

