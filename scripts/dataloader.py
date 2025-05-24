import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import create_sequence

def create_dataloader(dataset: np.array, batch_size: int, seq_length: int, target_dim: int) -> DataLoader:
    
    # tensor_dataset = TensorDataset(dataset)
    dataloader = DataLoader(torch.tensor(dataset).to(torch.float32), batch_size=batch_size, num_workers=4)
    
    return dataloader