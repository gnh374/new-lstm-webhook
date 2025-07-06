import subprocess
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn  
import asyncio
import aiohttp  
from flask import Flask, jsonify
from fetch_cpu import get_all_cpu_usage, get_all_cpu_usage_new
from scripts.dataloader import create_dataloader
from scripts.model import Predictor
from scripts.preprocessing import scale, Preprocessor 

WEBHOOK_ENDPOINTS = [
    "http://3.209.228.88:30080/api/trigger",
    "http://23.21.75.247:30080/api/trigger",
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
    
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        try:
          
            df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])

            preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)
            processed_data = preprocessor.transform(df['mean_CPU_usage_rate'].values, df['mean_CPU_usage_rate'].values)
            
            if isinstance(processed_data, torch.Tensor):
                if len(processed_data.shape) == 3:
       
                    input_data = processed_data
                   
                    if input_data.size(-1) != len(features):
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {input_data.size(-1)} to {len(features)}")
                        if input_data.size(-1) > len(features):
                            input_data = input_data[:, :, :len(features)]
                        else:
                            padding = torch.zeros(input_data.size(0), input_data.size(1), len(features) - input_data.size(-1), device=input_data.device)
                            input_data = torch.cat([input_data, padding], dim=-1)
                elif len(processed_data.shape) == 4:
                    print(f"Warning: Got 4D tensor with shape {processed_data.shape}, reducing to 3D")
                    input_data = processed_data.squeeze(-1)
                    if input_data.size(-1) != len(features):
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {input_data.size(-1)} to {len(features)}")
                        if input_data.size(-1) > len(features):
                            input_data = input_data[:, :, :len(features)]
                        else:
                            padding = torch.zeros(input_data.size(0), input_data.size(1), len(features) - input_data.size(-1), device=input_data.device)
                            input_data = torch.cat([input_data, padding], dim=-1)
                else:
                    if len(processed_data.shape) == 1:
                        input_data = processed_data.unsqueeze(0).unsqueeze(0)
                    else:
                        input_data = processed_data.unsqueeze(0)
                    
                    if input_data.size(-1) != len(features) and len(input_data.shape) >= 2:
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {input_data.size(-1)} to {len(features)}")
                        if input_data.size(-1) > len(features):
                            input_data = input_data[:, :, :len(features)]
                        else:
                            padding = torch.zeros(input_data.size(0), input_data.size(1), len(features) - input_data.size(-1), device=input_data.device)
                            input_data = torch.cat([input_data, padding], dim=-1)
                
                raw_prediction = model(input_data)
            else:
                df = processed_data
                feature_scaler = MinMaxScaler()
                
                feature_values = df[features].values
                feature_scaler.fit(feature_values)
                
                scaled_features = feature_scaler.transform(feature_values)
                
                seq = scaled_features[-lookback_window:]
                
                input_data = torch.tensor(seq).to(torch.float32)
                
                raw_prediction = model(input_data.unsqueeze(0))
            
            original_min = df["mean_CPU_usage_rate"].min()
            original_max = df["mean_CPU_usage_rate"].max()
            
            scaled_back_prediction = raw_prediction.item() * (original_max - original_min) + original_min
            
            return cluster_name, scaled_back_prediction
        except Exception as e:
            print(f"Error in predict_cpu_new: {e}")
            return cluster_name, 0.5

async def predict_cpu(cluster_name, data):
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])
        preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)
        df = preprocessor.transform(df['mean_CPU_usage_rate'].values, df['mean_CPU_usage_rate'].values)

        seq = df[features].tail(lookback_window).values

        input_data = torch.tensor(seq).to(torch.float32)
        prediction = model(input_data.unsqueeze(-1))

        return cluster_name, prediction

async def send_webhook_request(cluster_index):
    url = WEBHOOK_ENDPOINTS[cluster_index]  
    payload = {"selected_cluster": cluster_index}

    try:
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status >= 200 and response.status < 300:
                    return await response.text()
                else:
                    return f"Error: Received status code {response.status}"
    except aiohttp.ClientConnectorError as e:
        return f"Connection error to {url}: {str(e)}"
    except aiohttp.ClientError as e:
        return f"Request error to {url}: {str(e)}"
    except asyncio.TimeoutError:
        return f"Timeout error when connecting to {url}"
    except Exception as e:
        return f"Unexpected error when sending webhook to {url}: {str(e)}"
        
@app.route("/predict", methods=["POST"])
async def predict():
    try:
        cpu_data = await get_all_cpu_usage()
        print("CPU data received:", cpu_data)
        
        tasks = [predict_cpu_new(name, data) for name, data in cpu_data.items()]
        predictions = await asyncio.gather(*tasks)
        print("Predictions:", predictions)
        
        cpu_utilization = []
        for name, prediction in predictions:
            idx = int(name) if isinstance(name, str) and name.isdigit() else name
            if isinstance(prediction, torch.Tensor):
                normalized_value = prediction.item() / CPU_MAX[idx]
            else:
                normalized_value = prediction / CPU_MAX[idx]
            cpu_utilization.append((name, normalized_value))

        best_cluster = min(cpu_utilization, key=lambda x: x[1])
        best_cluster_name = best_cluster[0]
        best_cluster_index = int(best_cluster_name) if isinstance(best_cluster_name, str) and best_cluster_name.isdigit() else best_cluster_name
      
        print("Best cluster index:", best_cluster_index)
        webhook_response = await send_webhook_request(best_cluster_index)
        
        predictions_dict = {}
        for key, value in cpu_utilization:
            if isinstance(value, torch.Tensor):
                predictions_dict[key] = value.item()
            else:
                predictions_dict[key] = value
                
        return jsonify({
            "predictions": predictions_dict,
            "best_cluster": best_cluster_name,
            "webhook_response": webhook_response
        })
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict-terraform", methods=["POST"])
async def predict_combined():
    cpu_data = await get_all_cpu_usage_new()
    
    down_cluster_data = cpu_data.pop("down_cluster", None)
    
    all_tasks = []
    for name, data in cpu_data.items():
        all_tasks.append(predict_recursive(name, data))
    
    down_predictions = None
    if down_cluster_data:
        if "values" in down_cluster_data:
            down_cluster_values = down_cluster_data["values"]
            down_predictions = await predict_recursive("down_cluster", down_cluster_values)
    
    all_predictions = await asyncio.gather(*all_tasks)
    
    combined_predictions = []
    for name, predictions in all_predictions:
        if down_predictions:
            down_prediction_values = down_predictions[1]
            print("===========")
            print(down_predictions)
            print(predictions)
            predictions = [p + float(down_prediction_values[i]) for i, p in enumerate(predictions)]
            print("afterr adding down cluster ", predictions)
        max_prediction = max(predictions)
        combined_predictions.append((name, max_prediction))
    
    normalized_predictions = []
    for name, max_prediction in combined_predictions:
        idx = int(name) if isinstance(name, str) and name.isdigit() else name
        normalized_value = max_prediction / CPU_MAX[idx]
        normalized_predictions.append((name, normalized_value))
    
    best_cluster = min(normalized_predictions, key=lambda x: x[1])
    best_cluster_value = best_cluster[1]
    best_cluster_name = best_cluster[0]
    terraform_response = None
    webhook_response = None
    if best_cluster_value >= 0.8:
        terraform_response = run_terraform()
        print("Terraform applied successfully")
    else:
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
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))
    preprocessor = Preprocessor(lookback_window=4, window_length=5, polyorder=3)

    try:
        if isinstance(data, dict) and "cpu_loads" in data:
            cpu_loads = data["cpu_loads"]
        elif isinstance(data, list):
            if isinstance(data[0], dict) and "mean_CPU_usage_rate" in data[0]:
                cpu_loads = [item["mean_CPU_usage_rate"] for item in data]
            else:
                cpu_loads = data
        else:
            raise ValueError(f"Unknown data format: {type(data)}")

        required_length = 14
        if len(cpu_loads) < required_length:
            print(f"Warning: Not enough data points. Padding from {len(cpu_loads)} to {required_length}")
            if len(cpu_loads) > 0:
                padding = [cpu_loads[-1]] * (required_length - len(cpu_loads))
                cpu_loads = cpu_loads + padding
            else:
                cpu_loads = [0.5] * required_length

        scaler = MinMaxScaler()
        cpu_loads = scaler.fit_transform(np.array(cpu_loads).reshape(-1, 1)).flatten()

        predictions = []
        for _ in range(3):
            inputs = preprocessor.transform(cpu_loads, cpu_loads)
            if isinstance(inputs, torch.Tensor):
                if len(inputs.shape) == 3:
                    if inputs.size(-1) != len(features):
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {inputs.size(-1)} to {len(features)}")
                        if inputs.size(-1) > len(features):
                            inputs = inputs[:, :, :len(features)]
                        else:
                            padding = torch.zeros(inputs.size(0), inputs.size(1), len(features) - inputs.size(-1), device=inputs.device)
                            inputs = torch.cat([inputs, padding], dim=-1)
                else:
                    inputs = inputs.unsqueeze(0)
                    if inputs.size(-1) != len(features) and len(inputs.shape) >= 2:
                        print(f"Warning: Feature dimension mismatch. Reshaping tensor from {inputs.size(-1)} to {len(features)}")
                        if inputs.size(-1) > len(features):
                            inputs = inputs[:, :, :len(features)]
                        else:
                            padding = torch.zeros(inputs.size(0), inputs.size(1), len(features) - inputs.size(-1), device=inputs.device)
                            inputs = torch.cat([inputs, padding], dim=-1)
            else:
                df = inputs
                
                missing_features = [feat for feat in features if feat not in df.columns]
                if missing_features:
                    print(f"Warning: Missing features in DataFrame: {missing_features}")
                    for feat in missing_features:
                        df[feat] = 0.0
                
                feature_values = df[features].values
                
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(feature_values)
                scaled_features = feature_scaler.transform(feature_values)
                
                if len(scaled_features) < lookback_window:
                    print(f"Warning: Not enough rows for lookback window. Padding from {len(scaled_features)} to {lookback_window}")
                    padding = np.zeros((lookback_window - len(scaled_features), len(features)))
                    scaled_features = np.vstack([padding, scaled_features])
                
                seq = scaled_features[-lookback_window:]
                
                inputs = torch.tensor(seq).unsqueeze(0).to(torch.float32)

            model.eval()
            with torch.inference_mode():
                raw_prediction = model(inputs).squeeze().item()
            if hasattr(scaler, 'data_max_') and hasattr(scaler, 'data_min_'):
                denormalized_prediction = raw_prediction * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
            else:
                denormalized_prediction = raw_prediction

            predictions.append(float(denormalized_prediction))

            cpu_loads = np.append(cpu_loads[1:], raw_prediction)

        return cluster_name, predictions
    except Exception as e:
        print(f"Error in predict_recursive: {e}")
        return cluster_name, [0.5, 0.5]
def run_terraform():
    try:
        terraform_dir = "./"
        os.chdir(terraform_dir)

        init_process = subprocess.run(["terraform", "init"], capture_output=True, text=True)
        if init_process.returncode != 0:
            return {"error": "Terraform init failed", "details": init_process.stderr}

        apply_process = subprocess.run(["terraform", "apply", "-auto-approve"], capture_output=True, text=True)
        if apply_process.returncode != 0:
            return {"error": "Terraform apply failed", "details": apply_process.stderr}
        
        print("Getting Terraform state...")
        state_process = subprocess.run(["terraform", "show"], capture_output=True, text=True)
        
        print("Terraform State:")
        print(state_process.stdout)
        
        if os.path.exists("terraform.tfstate"):
            print("Reading raw terraform.tfstate file...")
            with open('terraform.tfstate', 'r') as state_file:
                state_content = state_file.read()
                print("Raw Terraform State File (first 500 chars):")
                print(state_content[:500])
        
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