import subprocess
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn  # Import nn module from torch
import asyncio
import aiohttp  # Tambah aiohttp untuk request async
from flask import Flask, jsonify
from fetch_cpu import get_all_cpu_usage, get_all_cpu_usage_new
from scripts.dataloader import create_dataloader
from scripts.model import Predictor
from scripts.preprocessing import scale, Preprocessor # Import dari fetch_cpu.py

WEBHOOK_ENDPOINTS = [
    "http://3.229.64.47:30080/api/trigger",
    "http://34.193.188.155:30080/api/trigger",
    "http://54.162.8.214:30080/api/trigger",
]

CPU_MAX = [
    3,3,3
]
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

async def predict_cpu_new(cluster_name, data):
    # Use hidden_size=128 to match the saved model dimensions
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        try:
            # Create DataFrame from input data
            df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])

            # Use the new Preprocessor class
            preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)
            processed_data = preprocessor.transform(df['mean_CPU_usage_rate'].values, df['mean_CPU_usage_rate'].values)
              # Check if the preprocessor returns a tensor or DataFrame
            if isinstance(processed_data, torch.Tensor):
                # If tensor, handle appropriately
                # Make sure the dimensions are correct
                if len(processed_data.shape) == 3:
                    # Shape is likely (batch, seq_len, features)
                    input_data = processed_data
                    # Check if the last dimension matches the expected input size
                    if input_data.size(-1) != len(features):
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {input_data.size(-1)} to {len(features)}")
                        # Reshape to match expected input size
                        if input_data.size(-1) > len(features):
                            # Truncate to the first len(features) dimensions
                            input_data = input_data[:, :, :len(features)]
                        else:
                            # Pad with zeros to match len(features)
                            padding = torch.zeros(input_data.size(0), input_data.size(1), len(features) - input_data.size(-1), device=input_data.device)
                            input_data = torch.cat([input_data, padding], dim=-1)
                elif len(processed_data.shape) == 4:
                    # If we get a 4D tensor, reduce it to 3D
                    print(f"Warning: Got 4D tensor with shape {processed_data.shape}, reducing to 3D")
                    # Typically, the shape would be [batch, seq_len, features, 1] or similar
                    # We need [batch, seq_len, features]
                    input_data = processed_data.squeeze(-1)
                    if input_data.size(-1) != len(features):
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {input_data.size(-1)} to {len(features)}")
                        if input_data.size(-1) > len(features):
                            input_data = input_data[:, :, :len(features)]
                        else:
                            padding = torch.zeros(input_data.size(0), input_data.size(1), len(features) - input_data.size(-1), device=input_data.device)
                            input_data = torch.cat([input_data, padding], dim=-1)
                else:
                    # If it's 1D or 2D, add necessary dimensions
                    if len(processed_data.shape) == 1:
                        # Add batch and sequence dimensions
                        input_data = processed_data.unsqueeze(0).unsqueeze(0)
                    else:  # 2D
                        # Add batch dimension
                        input_data = processed_data.unsqueeze(0)
                    
                    # Check if we need to reshape the tensor
                    if input_data.size(-1) != len(features) and len(input_data.shape) >= 2:
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {input_data.size(-1)} to {len(features)}")
                        if input_data.size(-1) > len(features):
                            input_data = input_data[:, :, :len(features)]
                        else:
                            padding = torch.zeros(input_data.size(0), input_data.size(1), len(features) - input_data.size(-1), device=input_data.device)
                            input_data = torch.cat([input_data, padding], dim=-1)
                
                # Make prediction with the correctly shaped tensor
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

async def predict_cpu(cluster_name, data):
    # Use hidden_size=128 to match the saved model dimensions
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])        # Use the new Preprocessor class
        preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)
        df = preprocessor.transform(df['mean_CPU_usage_rate'].values, df['mean_CPU_usage_rate'].values)

        # feature_scaler = MinMaxScaler()
    
        # cpu_usage_data = scale(df.copy(), features, feature_scaler)
        
        # cpu_usage_data = create_dataloader(np.array(cpu_usage_data), 63, lookback_window, 1)
        seq = df[features].tail(lookback_window).values

        input_data = torch.tensor(seq).to(torch.float32)
        prediction = model(input_data.unsqueeze(-1))

        return cluster_name, prediction

async def send_webhook_request(cluster_index):
    url = WEBHOOK_ENDPOINTS[cluster_index]  
    payload = {"selected_cluster": cluster_index}

    try:
        async with aiohttp.ClientSession() as session:
            # Set a timeout for the request
            timeout = aiohttp.ClientTimeout(total=10)  # 10 seconds timeout
            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status >= 200 and response.status < 300:
                    return await response.text()
                else:
                    return f"Error: Received status code {response.status}"
    except aiohttp.ClientConnectorError as e:
        # Handle connection errors specifically
        return f"Connection error to {url}: {str(e)}"
    except aiohttp.ClientError as e:
        # Handle other aiohttp errors
        return f"Request error to {url}: {str(e)}"
    except asyncio.TimeoutError:
        # Handle timeout errors
        return f"Timeout error when connecting to {url}"
    except Exception as e:
        # Catch all other exceptions
        return f"Unexpected error when sending webhook to {url}: {str(e)}"
        
@app.route("/predict", methods=["POST"])
async def predict():
    try:
        cpu_data = await get_all_cpu_usage()  # Get CPU data from all clusters
        print("CPU data received:", cpu_data)
        
        tasks = [predict_cpu_new(name, data) for name, data in cpu_data.items()]
        predictions = await asyncio.gather(*tasks)
        print("Predictions:", predictions)
        
        # Normalize predictions by dividing by CPU_MAX
        cpu_utilization = []
        for name, prediction in predictions:
            # Convert name to integer if it's a string index
            idx = int(name) if isinstance(name, str) and name.isdigit() else name
            # Normalize by dividing by corresponding CPU_MAX value
            if isinstance(prediction, torch.Tensor):
                # If prediction is a tensor, convert to a Python float
                normalized_value = prediction.item() / CPU_MAX[idx]
            else:
                # If prediction is already a Python number
                normalized_value = prediction / CPU_MAX[idx]
            cpu_utilization.append((name, normalized_value))

        # Find cluster with lowest CPU usage prediction (normalized)
        best_cluster = min(cpu_utilization, key=lambda x: x[1])
        best_cluster_name = best_cluster[0]  # Get the cluster name
        best_cluster_index = int(best_cluster_name) if isinstance(best_cluster_name, str) and best_cluster_name.isdigit() else best_cluster_name
      
        print("Best cluster index:", best_cluster_index)
        webhook_response = await send_webhook_request(best_cluster_index)
        
        # Convert any tensor values to Python native types for JSON serialization
        predictions_dict = {}
        for key, value in cpu_utilization:
            if isinstance(value, torch.Tensor):
                predictions_dict[key] = value.item()
            else:
                predictions_dict[key] = value
                
        return jsonify({
            "predictions": predictions_dict,
            "best_cluster": best_cluster_name,
            "webhook_response": webhook_response  # Include webhook response
        })
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route("/predict-combined", methods=["POST"])
# async def predict_combined():
#     # Fetch CPU data from all clusters
#     cpu_data = await get_all_cpu_usage_new()
    
#     # Extract the down cluster data but keep a copy
#     down_cluster_data = cpu_data.pop("down_cluster", None)
    
#     # First predict CPU usage for all clusters individually
#     all_tasks = []
    
#     # Add tasks for active clusters
#     for name, data in cpu_data.items():
#         all_tasks.append(predict_cpu_new(name, data))
    
#     # Add task for down cluster if it exists
#     down_prediction = None
#     if down_cluster_data and "values" in down_cluster_data:
#         all_tasks.append(predict_cpu_new("down_cluster", down_cluster_data["values"]))
    
#     # Wait for all predictions
#     all_predictions = await asyncio.gather(*all_tasks)
    
#     # Separate down cluster prediction
#     active_predictions = []
#     for pred in all_predictions:
#         if pred[-1] == "down_cluster":
#             down_prediction = pred[0]
#         else:
#             active_predictions.append(pred)
    
#     # Add down cluster prediction to all other clusters' predictions
#     combined_predictions = []
#     if down_prediction is not None:
#         for name, prediction in active_predictions:
#             # Add down cluster prediction to this cluster's prediction
#             combined_predictions.append((name, prediction + down_prediction))
#     else:
#         combined_predictions = active_predictions
#     # Normalize predictions based on CPU_MAX values before finding minimum
#     cpu_utilization = []
#     for name, prediction in combined_predictions:
#         # Convert name to integer if it's a string index
#         idx = int(name) if isinstance(name, str) and name.isdigit() else name
#         # Normalize by dividing by corresponding CPU_MAX value
#         normalized_value = prediction / CPU_MAX[idx]
#         cpu_utilization.append((name, normalized_value))

    
#     # Find cluster with lowest normalized predicted CPU usage
#     best_cluster = min(cpu_utilization, key=lambda x: x[0])
#     best_cluster_value = best_cluster[0].item()  # Get the value and convert tensor to float
    
#     # Check if even the best cluster is highly utilized (>79%)
#     terraform_response = None
#     webhook_response = None

#     if best_cluster_value > -1.8:
#         # If all clusters are heavily loaded, create new resources with Terraform
#         terraform_response = run_terraform()
#         print("Terraform applied successfully")
#     else:
#         # Otherwise, send webhook request to the best cluster
#         webhook_response = await send_webhook_request(best_cluster[-1])
#         print("best cluster: ", best_cluster[-1])
#     return jsonify({
#         "predictions": {key: value.item() for key, value in combined_predictions},
#         "cpu_utilization": {key: value.item() for key, value in cpu_utilization},
#         "original_predictions": {key: value.item() for key, value in active_predictions},
#         "down_cluster_prediction": down_prediction.item() if down_prediction is not None else None,
#         "best_cluster": best_cluster[-1],
#         "best_cluster_utilization": best_cluster_value,
#         "terraform_applied": terraform_response is not None,
#         "terraform_response": terraform_response,
#         "webhook_response": webhook_response,
#         "down_cluster_status": down_cluster_data.get("status") if down_cluster_data else "unknown"
#     })

@app.route("/predict-terraform", methods=["POST"])
async def predict_combined():
    # Fetch CPU data from all clusters
    cpu_data = await get_all_cpu_usage_new()
    
    # Extract the down cluster data
    down_cluster_data = cpu_data.pop("down_cluster", None)
    
    # Predict recursively for all clusters
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
    best_cluster_name = best_cluster[0]  # Get the cluster name    # Check if even the best cluster is highly utilized (>79%)
    terraform_response = None
    webhook_response = None
    if best_cluster_value >= 0.8:  # Using 79% utilization as threshold
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
        "best_cluster_utilization": best_cluster_value,
        "terraform_applied": terraform_response is not None,
        "terraform_response": terraform_response,
        "webhook_response": webhook_response,
        "down_cluster_status": down_cluster_data.get("status") if down_cluster_data else "unknown"
    })
    

async def predict_recursive(cluster_name, data):
    # Use hidden_size=128 to match the saved model dimensions
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))
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

        # Recursive prediction for 2 steps ahead
        predictions = []
        for _ in range(3):
            # Preprocess and get the last sequence
            inputs = preprocessor.transform(cpu_loads, cpu_loads)  # Only CPU loads are used            # Check the shape of inputs
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
                raw_prediction = model(inputs).squeeze().item()            # Denormalize the prediction
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
    try:
        # Change directory to where the Terraform files are located
        terraform_dir = "./"  # Adjust this path if needed
        os.chdir(terraform_dir)

        # Run `terraform init`
        init_process = subprocess.run(["terraform", "init"], capture_output=True, text=True)
        if init_process.returncode != 0:  # Changed from -1 to 0
            return {"error": "Terraform init failed", "details": init_process.stderr}

        # Run `terraform apply` with auto-approve
        apply_process = subprocess.run(["terraform", "apply", "-auto-approve"], capture_output=True, text=True)
        if apply_process.returncode != 0:  # Changed from -1 to 0
            return {"error": "Terraform apply failed", "details": apply_process.stderr}
        
        # Get the terraform state
        print("Getting Terraform state...")
        state_process = subprocess.run(["terraform", "show"], capture_output=True, text=True)
        
        # Print state to container logs
        print("Terraform State:")
        print(state_process.stdout)
        
        # Check if the state file exists and read it
        if os.path.exists("terraform.tfstate"):
            print("Reading raw terraform.tfstate file...")
            with open('terraform.tfstate', 'r') as state_file:
                state_content = state_file.read()
                print("Raw Terraform State File (first 500 chars):")
                print(state_content[:500])  # Print just the beginning to avoid huge logs
        
        return {
            "message": "Terraform applied successfully", 
            "output": apply_process.stdout,
            "state": state_process.stdout
        }

    except Exception as e:
        print(f"Error in run_terraform: {e}")
        return {"error": str(e)}


 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)