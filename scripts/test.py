from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

def test(model: nn.Module, test_set: DataLoader):
    all_truth = []
    all_pred = []

    with torch.inference_mode():
        model.eval()
        for X_batch_test, y_batch_test in test_set: 
            test_pred = model(X_batch_test)

            all_pred.extend(test_pred.numpy())
            all_truth.extend(y_batch_test.numpy())

    print(f"RMSE: {np.sqrt(mean_squared_error(all_truth, all_pred))}")
    print(f"RMSLE: {root_mean_squared_log_error(all_truth, all_pred)}")
    print(f"MAE: {mean_absolute_error(all_truth, all_pred)}")
    print(f"MAPE: {mean_absolute_percentage_error(all_truth, all_pred)}")
    print(f"R2 Score: {r2_score(all_truth, all_pred)}")

    plt.plot(all_truth, label="Actual")
    plt.plot(all_pred, label="Prediction", linestyle="dashed")

    plt.title("LSTM Predictions Over Time") 
    plt.show()