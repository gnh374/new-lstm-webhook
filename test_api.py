import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_predict_endpoint():
    print("Testing /predict endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/predict", json={})
        if response.status_code == 200:
            print("✓ Success!")
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"✗ Failed with status code {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")

def test_predict_terraform_endpoint():
    print("\nTesting /predict-terraform endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/predict-terraform", json={})
        if response.status_code == 200:
            print("✓ Success!")
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"✗ Failed with status code {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("Waiting for server to start...")
    time.sleep(2)
    
    test_predict_endpoint()
    test_predict_terraform_endpoint()
    
    print("\nTests completed.")
