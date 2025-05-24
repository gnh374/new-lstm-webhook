"""
Mock implementation of fetch_cpu.py to be used for testing 
without real connections to external APIs.
"""
import json
import asyncio

# Load mock data
def load_mock_data():
    try:
        with open('mock_data.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading mock data: {e}")
        # Return default data if file doesn't exist
        return {
            "cluster_data": {
                "0": [{"mean_CPU_usage_rate": 0.5} for _ in range(15)],
                "1": [{"mean_CPU_usage_rate": 0.7} for _ in range(15)],
                "2": [{"mean_CPU_usage_rate": 0.3} for _ in range(15)]
            },
            "down_cluster_data": {
                "status": "down",
                "values": [{"mean_CPU_usage_rate": 0.4} for _ in range(15)]
            }
        }

async def get_all_cpu_usage():
    """Mock function for get_all_cpu_usage"""
    # Simulate some delay to mimic network call
    await asyncio.sleep(0.5)
    
    mock_data = load_mock_data()
    return {
        "0": mock_data["cluster_data"]["0"],
        "1": mock_data["cluster_data"]["1"],
        "2": mock_data["cluster_data"]["2"]
    }

async def get_all_cpu_usage_new():
    """Mock function for get_all_cpu_usage_new"""
    # Simulate some delay to mimic network call
    await asyncio.sleep(0.5)
    
    mock_data = load_mock_data()
    result = {
        "0": {"cpu_loads": [item["mean_CPU_usage_rate"] for item in mock_data["cluster_data"]["0"]]},
        "1": {"cpu_loads": [item["mean_CPU_usage_rate"] for item in mock_data["cluster_data"]["1"]]},
        "2": {"cpu_loads": [item["mean_CPU_usage_rate"] for item in mock_data["cluster_data"]["2"]]},
    }
    
    # Add down_cluster data if available
    if "down_cluster_data" in mock_data:
        result["down_cluster"] = {
            "status": mock_data["down_cluster_data"]["status"],
            "values": {"cpu_loads": [item["mean_CPU_usage_rate"] for item in mock_data["down_cluster_data"]["values"]]}
        }
    
    return result
