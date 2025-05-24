import torch
from torch import nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    
def create_sequence(data: torch.Tensor, seq_length: int, output_dim: int):
    num_samples = data.shape[0] - seq_length
    
    X = torch.stack([data[i : i + seq_length, output_dim:] for i in range(num_samples)])
    Y = data[seq_length:, 0:output_dim]
    
    return X, Y