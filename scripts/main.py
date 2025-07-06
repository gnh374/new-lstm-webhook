import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.optim import AdamW

from model import Predictor
from utils import RMSLELoss
from preprocessing import preprocess, scale, split_dataset
from dataloader import create_dataloader
from train import train
from test import test

if __name__ == '__main__':
    lookback_window = 5 

    df = pd.read_csv('../processed/actually_binned_sum_2_task_usage.csv')

    df = preprocess(df, lookback_window, 2)
     
    features = [
        'rolling_std_CPU_usage',
        'rolling_mean_CPU_usage',
        'transformed_mean_CPU_usage_rate',
        *[f'mean_CPU_usage_rate_-{i*2}min' for i in range(1, lookback_window + 1)],

       
    ]

    feature_scaler = MinMaxScaler()

    
    train_set = scale(df.copy(), features, feature_scaler)
    
    train_set = create_dataloader(np.array(train_set), 64, lookback_window, 1)
   
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("../models/best_model.pt"))
    model.eval()

    y_prd = model(train_set)

    