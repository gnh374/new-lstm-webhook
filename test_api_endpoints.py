import requests
import json
import traceback
import time

# Define the base URL of the API
BASE_URL = "http://127.0.0.1:8000"

def test_predict_endpoint():
    """Test the /predict endpoint"""
    print("\n=== Testing /predict endpoint ===")
    try:
        # Add a timeout to prevent hanging
        response = requests.post(f"{BASE_URL}/predict", json={}, timeout=10)
        if response.status_code == 200:
            print("✓ Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.Timeout:
        print("✗ Request timed out. Server may be processing a long task.")
    except requests.exceptions.ConnectionError:
        print("✗ Connection error. Make sure the server is running.")
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        print(traceback.format_exc())

def test_predict_terraform_endpoint():
    """Test the /predict-terraform endpoint"""
    print("\n=== Testing /predict-terraform endpoint ===")
    try:
        # Add a timeout to prevent hanging
        response = requests.post(f"{BASE_URL}/predict-terraform", json={}, timeout=10)
        if response.status_code == 200:
            print("✓ Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.Timeout:
        print("✗ Request timed out. Server may be processing a long task.")
    except requests.exceptions.ConnectionError:
        print("✗ Connection error. Make sure the server is running.")
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Running API tests...")
    
    # Make sure the server is up before testing
    time.sleep(2)
    
    # Test both endpoints
    test_predict_endpoint()
    test_predict_terraform_endpoint()
    
    print("\nTests completed.")
