import torch
import torch.nn as nn
import torch.nn.functional as f

class BasicTaxoModel(nn.Module):
    def __init__(self,
                 input_size: int, # kmer size
                 hidden_size: int,
                 output_size: int = 128, # num_classes
                 ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        return x
