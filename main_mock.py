"""
Modified version of main.py for testing.
This uses the mock fetch_cpu implementation to avoid the need for external connections.
"""
import subprocess
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import asyncio
from flask import Flask, jsonify, request

# Import the mock implementation instead of the real one
from fetch_cpu_mock import get_all_cpu_usage, get_all_cpu_usage_new
from scripts.dataloader import create_dataloader
from scripts.model import Predictor
from scripts.preprocessing import Preprocessor

# Constants
WEBHOOK_ENDPOINTS = [
    "http://54.88.29.103:30080/api/trigger",
    "http://54.162.198.32:30080/api/trigger",
    "http://98.85.113.75:30080/api/trigger",
]

CPU_MAX = [3, 3, 3]  # Match main.py
lookback_window = 4
lag_features = []

for i in range(0, lookback_window + 1):
    lag_features.append(f'mean_CPU_usage_rate_-{i*1}min')

features = [
    'transformed_mean_CPU_usage_rate',
    'rolling_std_CPU_usage',
    'rolling_mean_CPU_usage',
    *lag_features,
]

app = Flask(__name__)

# Mock model to avoid needing the actual model file
class MockModel:
    def __init__(self):
        pass
        
    def eval(self):
        return self
        
    def __call__(self, x):
        # Return a consistent prediction for testing
        batch_size = x.size(0) if len(x.shape) > 1 else 1
        return torch.tensor([0.8] * batch_size)

async def predict_cpu_new(cluster_name, data):
    # Use mock model if best_model.pt doesn't exist
    try:
        model = Predictor(len(features), 128, 1)
        model.load_state_dict(torch.load("./best_model.pt"))
    except:
        print("Using mock model")
        model = MockModel()

    with torch.inference_mode():
        model.eval()
        try:
            # Create DataFrame from input data
            df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])

            # Use the Preprocessor class
            preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)
            processed_data = preprocessor.transform(df['mean_CPU_usage_rate'].values, df['mean_CPU_usage_rate'].values)
            
            # Check if the preprocessor returns a tensor or DataFrame
            if isinstance(processed_data, torch.Tensor):
                # If tensor, reshape for model input
                input_data = processed_data.unsqueeze(0)  # Add batch dimension
                raw_prediction = model(input_data)
            else:
                # If DataFrame, extract features, scale, and convert to tensor
                df = processed_data
                feature_scaler = MinMaxScaler()
                
                # Fit scaler on the features we're using
                feature_values = df[features].values
                feature_scaler.fit(feature_values)
                
                # Scale the features
                scaled_features = feature_scaler.transform(feature_values)
                
                # Get the last lookback_window rows for model input
                seq = scaled_features[-lookback_window:]
                
                # Convert to tensor for model
                input_data = torch.tensor(seq).to(torch.float32)
                
                # Make prediction
                raw_prediction = model(input_data.unsqueeze(0))  # Add batch dimension
            
            # Get original data stats for scaling back
            original_min = df["mean_CPU_usage_rate"].min()
            original_max = df["mean_CPU_usage_rate"].max()
            
            # Scale back prediction
            scaled_back_prediction = raw_prediction.item() * (original_max - original_min) + original_min
            
            return cluster_name, scaled_back_prediction
        except Exception as e:
            print(f"Error in predict_cpu_new: {e}")
            # Return a default prediction in case of error
            return cluster_name, 0.5

async def send_webhook_request(cluster_index):
    # Mock webhook response
    print(f"Mock webhook request to cluster {cluster_index}")
    return f"Success from cluster {cluster_index}"

async def predict_recursive(cluster_name, data):
    """
    Real implementation of predict_recursive that uses the actual model for predictions.
    This will help test the model's prediction capabilities with consistent data.
    """
    # Use hidden_size=128 to match the saved model dimensions
    try:
        model = Predictor(len(features), 128, 1)
        model.load_state_dict(torch.load("./best_model.pt"))
    except Exception as e:
        print(f"Could not load model, using mock: {e}")
        model = MockModel()
        
    preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)

    try:
        # Extract CPU loads, handling different possible data formats
        if isinstance(data, dict) and "cpu_loads" in data:
            cpu_loads = data["cpu_loads"]
        elif isinstance(data, list):
            # Handle case where data is a list directly
            if isinstance(data[0], dict) and "mean_CPU_usage_rate" in data[0]:
                # List of dictionaries with mean_CPU_usage_rate
                cpu_loads = [item["mean_CPU_usage_rate"] for item in data]
            else:
                # Assume it's a list of values
                cpu_loads = data
        else:
            # If we can't determine the format, raise an error
            raise ValueError(f"Unknown data format: {type(data)}")

        # Pad data if we don't have enough points
        required_length = 14
        if len(cpu_loads) < required_length:
            print(f"Warning: Not enough data points. Padding from {len(cpu_loads)} to {required_length}")
            # Pad by repeating the last value
            if len(cpu_loads) > 0:
                padding = [cpu_loads[-1]] * (required_length - len(cpu_loads))
                cpu_loads = cpu_loads + padding
            else:
                # If no data at all, use zeros
                cpu_loads = [0.5] * required_length

        # Normalize CPU loads
        scaler = MinMaxScaler()
        cpu_loads = scaler.fit_transform(np.array(cpu_loads).reshape(-1, 1)).flatten()

        # Recursive prediction for 3 steps ahead
        predictions = []
        for _ in range(3):
            # Preprocess and get the last sequence
            inputs = preprocessor.transform(cpu_loads, cpu_loads)  # Only CPU loads are used
            
            # Check the shape of inputs
            if isinstance(inputs, torch.Tensor):
                # If it's already a tensor, handle appropriately
                # Make sure the dimensions are correct
                if len(inputs.shape) == 3:
                    # Shape is likely (batch, seq_len, features)
                    # Check if the last dimension matches the expected input size
                    if inputs.size(-1) != len(features):
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {inputs.size(-1)} to {len(features)}")
                        # Reshape to match expected input size
                        # We'll either truncate or pad the features dimension
                        if inputs.size(-1) > len(features):
                            # Truncate to the first len(features) dimensions
                            inputs = inputs[:, :, :len(features)]
                        else:
                            # Pad with zeros to match len(features)
                            padding = torch.zeros(inputs.size(0), inputs.size(1), len(features) - inputs.size(-1), device=inputs.device)
                            inputs = torch.cat([inputs, padding], dim=-1)
                else:
                    # Add necessary dimensions if needed
                    inputs = inputs.unsqueeze(0)  # Add batch dimension if needed
                    # Check if we need to reshape the tensor
                    if inputs.size(-1) != len(features) and len(inputs.shape) >= 2:
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {inputs.size(-1)} to {len(features)}")
                        # Same reshaping logic as above
                        if inputs.size(-1) > len(features):
                            inputs = inputs[:, :, :len(features)]
                        else:
                            padding = torch.zeros(inputs.size(0), inputs.size(1), len(features) - inputs.size(-1), device=inputs.device)
                            inputs = torch.cat([inputs, padding], dim=-1)
            else:
                # If it's a DataFrame, handle differently
                df = inputs
                
                # Make sure we have all required features
                missing_features = [feat for feat in features if feat not in df.columns]
                if missing_features:
                    print(f"Warning: Missing features in DataFrame: {missing_features}")
                    # Add missing features with default values
                    for feat in missing_features:
                        df[feat] = 0.0
                
                # Now extract only the features we need
                feature_values = df[features].values
                
                # Scale the features
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(feature_values)
                scaled_features = feature_scaler.transform(feature_values)
                
                # Make sure we have enough rows for lookback window
                if len(scaled_features) < lookback_window:
                    print(f"Warning: Not enough rows for lookback window. Padding from {len(scaled_features)} to {lookback_window}")
                    # Pad with zeros
                    padding = np.zeros((lookback_window - len(scaled_features), len(features)))
                    scaled_features = np.vstack([padding, scaled_features])
                
                # Get the last lookback_window rows for model input
                seq = scaled_features[-lookback_window:]
                
                # Convert to tensor for model
                inputs = torch.tensor(seq).unsqueeze(0).to(torch.float32)

            model.eval()
            with torch.inference_mode():
                # Make prediction
                raw_prediction = model(inputs).squeeze().item()
                
            # Denormalize the prediction
            if hasattr(scaler, 'data_max_') and hasattr(scaler, 'data_min_'):
                denormalized_prediction = raw_prediction * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
            else:
                # Fallback if scaler attributes are not available
                denormalized_prediction = raw_prediction

            # Append the denormalized prediction
            predictions.append(float(denormalized_prediction))

            # Update the CPU loads for the next step
            cpu_loads = np.append(cpu_loads[1:], raw_prediction)

        return cluster_name, predictions
    except Exception as e:
        print(f"Error in predict_recursive: {e}")
        # Return default predictions in case of error
        return cluster_name, [0.5, 0.5]

def run_terraform():
    # Mock terraform execution that follows the structure in main.py
    print("Mocking Terraform execution...")
    try:
        # Return the same structure as the real function
        return {"message": "Terraform applied successfully (MOCK)", "output": "Mock terraform output"}
    except Exception as e:
        return {"error": str(e)}

@app.route("/predict", methods=["POST"])
async def predict():
    try:
        cpu_data = await get_all_cpu_usage()  # Get CPU data from mock
        tasks = [predict_cpu_new(name, data) for name, data in cpu_data.items()]
        predictions = await asyncio.gather(*tasks)
        
        # Normalize predictions by dividing by CPU_MAX
        cpu_utilization = []
        for name, prediction in predictions:
            # Convert name to integer if it's a string index
            idx = int(name) if isinstance(name, str) and name.isdigit() else name
            # Normalize by dividing by corresponding CPU_MAX value
            normalized_value = prediction / CPU_MAX[idx]
            cpu_utilization.append((name, normalized_value))

        # Find cluster with lowest CPU usage prediction
        best_cluster = min(cpu_utilization, key=lambda x: x[1])
        best_cluster_index = int(best_cluster[0])
        
        webhook_response = await send_webhook_request(best_cluster_index)

        return jsonify({
            "predictions": {str(key): float(value) for key, value in predictions},
            "cpu_utilization": {str(key): float(value) for key, value in cpu_utilization},
            "best_cluster": best_cluster_index,
            "webhook_response": webhook_response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-terraform", methods=["POST"])
async def predict_combined():
    try:
        # Get mock CPU data
        cpu_data = await get_all_cpu_usage_new()
        
        # Extract the down cluster data
        down_cluster_data = cpu_data.pop("down_cluster", None)
        
        # Use predict_recursive to get all recursive predictions
        all_tasks = []
        for name, data in cpu_data.items():
            all_tasks.append(predict_recursive(name, data))
        
        # Predict recursively for the down cluster if it exists
        down_predictions = None
        if down_cluster_data:
            if "values" in down_cluster_data:
                # The values could be in different formats
                down_cluster_values = down_cluster_data["values"]
                down_predictions = await predict_recursive("down_cluster", down_cluster_values)
        
        # Wait for all predictions
        all_predictions = await asyncio.gather(*all_tasks)
        
        # Combine predictions with the down cluster predictions
        combined_predictions = []
        for name, predictions in all_predictions:
            # Add the down cluster predictions to the current cluster's predictions
            if down_predictions:
                # Extract the list of predictions from down_predictions
                down_prediction_values = down_predictions[1]  # Using index 1 as it contains the predictions list
                # Add each down cluster prediction to the corresponding current cluster prediction
                print("===========")
                print(down_predictions)
                print(predictions)
                predictions = [p + float(down_prediction_values[i]) for i, p in enumerate(predictions)]
                print("afterr adding down cluster ", predictions)
            # Get the maximum prediction for the cluster
            max_prediction = max(predictions)
            combined_predictions.append((name, max_prediction))
            
        # Normalize predictions based on CPU_MAX
        normalized_predictions = []
        for name, max_prediction in combined_predictions:
            idx = int(name) if isinstance(name, str) and name.isdigit() else name
            normalized_value = max_prediction / CPU_MAX[idx]
            normalized_predictions.append((name, normalized_value))
        
        # Find the cluster with the minimum utilization
        best_cluster = min(normalized_predictions, key=lambda x: x[1])  # Use x[1] to get the value, not x[0]
        best_cluster_value = best_cluster[1]  # Get the normalized value
        best_cluster_name = best_cluster[0]  # Get the cluster name
        
        # Determine action based on utilization
        terraform_response = None
        webhook_response = None
        
        if best_cluster_value >= 0.79:  # Using 79% utilization as threshold
            # If all clusters are heavily loaded, create new resources with Terraform
            terraform_response = run_terraform()
            print("Terraform applied successfully")
        else:
            # Otherwise, send webhook request to the best cluster
            best_cluster_index = int(best_cluster_name) if isinstance(best_cluster_name, str) and best_cluster_name.isdigit() else best_cluster_name
            webhook_response = await send_webhook_request(best_cluster_index)
            print("Best cluster:", best_cluster_index)
        
        return jsonify({
            "predictions": {key: value for key, value in combined_predictions},
            "cpu_utilization": {key: value for key, value in normalized_predictions},
            "best_cluster": best_cluster_name,
            "best_cluster_utilization": float(best_cluster_value),
            "terraform_applied": terraform_response is not None,
            "terraform_response": terraform_response,
            "webhook_response": webhook_response,
            "down_cluster_status": down_cluster_data["status"] if down_cluster_data else "unknown"
        })
    except Exception as e:
        print(f"Error in predict_combined: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting mock API server on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=True)
