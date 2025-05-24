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
    
    #ubah data dari query prometheus jadi df dengan klm 'mean_CPU_usage_rate'
    #10menit per 2 menit, predict 2 menit
    df = pd.read_csv('../processed/actually_binned_sum_2_task_usage.csv')

    df = preprocess(df, lookback_window, 2)
     
    features = [
        'rolling_std_CPU_usage',
        'rolling_mean_CPU_usage',
        'transformed_mean_CPU_usage_rate',
        *[f'mean_CPU_usage_rate_-{i*2}min' for i in range(1, lookback_window + 1)],

        # 'rolling_std_memory_usage',
        # 'rolling_mean_memory_usage',
        # 'transformed_canonical_memory_usage',
        # *[f'canonical_memory_usage_-{i*2}min' for i in range(1, lookback_window + 1)],
    ]

    # target = [
    #     'mean_CPU_usage_rate',
    #     # 'canonical_memory_usage',
    # ]

    # df = df[target + features]
    
    feature_scaler = MinMaxScaler()
    # target_scaler = MinMaxScaler()
   
    # train_set, test_set = split_dataset(df)
    
    train_set = scale(df.copy(), features, feature_scaler)
    
    train_set = create_dataloader(np.array(train_set), 64, lookback_window, 1)
   
    # model = Predictor(len(features), 128, len(target))
    # optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("../models/best_model.pt"))
    model.eval()
    # best_model = train(model, RMSLELoss(), optimizer, 100, train_set, '../models/best_model.pth')

    y_prd = model(train_set)
    # test(best_model, test_set)
    
    