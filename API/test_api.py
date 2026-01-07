import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_prediction():
    """Test single prediction"""
    data = {
        "features": {
            "qty_dot_url": 2.0,
            "length_url": 45.0,
            "qty_hyphen_url": 1.0,
            # Add all your features here
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Single Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_prediction():
    """Test batch prediction"""
    data = {
        "samples": [
            {
                "features": {
                    "qty_dot_url": 2.0,
                    "length_url": 45.0,
                    "qty_hyphen_url": 1.0
                }
            },
            {
                "features": {
                    "qty_dot_url": 5.0,
                    "length_url": 120.0,
                    "qty_hyphen_url": 8.0
                }
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=data)
    print("Batch Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_get_features():
    """Test get features endpoint"""
    response = requests.get(f"{BASE_URL}/features")
    print("Available Features:")
    result = response.json()
    print(f"Feature count: {result['feature_count']}")
    print(f"Features: {result['features'][:10]}...")  # Show first 10
    print()

if __name__ == "__main__":
    print("Testing Phishing Detection API\n")
    test_health()
    test_get_features()
    test_single_prediction()
    test_batch_prediction()
