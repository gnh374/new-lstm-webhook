import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import torch
from sklearn.preprocessing import MinMaxScaler
import sys


class Preprocessor:
    def __init__(self, lookback_window: int = 5, window_length: int = 5, polyorder: int = 3):
        self.lookback_window = lookback_window
        self.window_length = window_length
        self.polyorder = polyorder
        
    def transform(self, cpu_loads: np.ndarray, memory_loads: np.ndarray) -> torch.tensor:
        # Generate all features
        features = self._generate_features(cpu_loads, memory_loads)
        
        # Scale all features
        return torch.from_numpy(features).float()[-self.window_length:].unsqueeze(0)
    
    def _generate_features(self, cpu_loads: np.ndarray, memory_loads: np.ndarray) -> np.ndarray:
        cpu = np.asarray(cpu_loads).squeeze()
        memory = np.asarray(memory_loads).squeeze()
        n_samples = cpu.shape[0]

        cpu_log = np.log1p(cpu)
        memory_log = np.log1p(memory)
        filtered_cpu = savgol_filter(cpu_log, window_length=self.window_length, polyorder=self.polyorder)
        filtered_memory = savgol_filter(memory_log, window_length=self.window_length, polyorder=self.polyorder)

        lagged = []
        for lag in range(1, self.lookback_window + 1):
            shifted_cpu = np.zeros(n_samples)
            shifted_memory = np.zeros(n_samples)
            if lag < n_samples:
                shifted_cpu[lag:] = cpu[:-lag]
                shifted_memory[lag:] = memory[:-lag]
            lagged.extend([shifted_cpu, shifted_memory])

        rolled = []
        if n_samples > 1:
            r_mean_cpu = np.zeros(n_samples)
            r_std_cpu = np.zeros(n_samples)
            r_mean_cpu[1:] = (cpu[1:] + cpu[:-1]) / 2
            r_std_cpu[1:] = np.abs(cpu[1:] - cpu[:-1]) / np.sqrt(2)
            r_mean_mem = np.zeros(n_samples)
            r_std_mem = np.zeros(n_samples)
            r_mean_mem[1:] = (memory[1:] + memory[:-1]) / 2
            r_std_mem[1:] = np.abs(memory[1:] - memory[:-1]) / np.sqrt(2)
            rolled = [r_std_cpu, r_std_mem, r_mean_cpu, r_mean_mem]
        else:
            rolled = [np.zeros(n_samples) for _ in range(4)]

        return np.column_stack([filtered_cpu, filtered_memory] + rolled + lagged)

    def save(self, path: str):
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path: str) -> 'Preprocessor':
        preprocessor = Preprocessor()
        preprocessor.__dict__ = torch.load(path, weights_only=False)
        return preprocessor

def split_dataset(df: pd.DataFrame):
    # Calculate the sizes for each split
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size

    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]
    
    return train_set, test_set
    

def scale(train_set: pd.DataFrame, features: list[str], feature_scaler) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set[features] = feature_scaler.fit_transform(train_set[features])
    
    return train_set

if __name__ == "main":
    fileName = sys.argv[1] 
    df = pd.read_csv(fileName)
    df = preprocess(df, 5, 2)
    print(df.head())