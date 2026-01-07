from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List

app = FastAPI(
    title="Phishing Detection API",
    version="1.0",
    description="API for detecting phishing URLs using deep learning"
)

# Define the PhishingNet model
class PhishingNet(nn.Module):
    def __init__(self, input_dim):
        super(PhishingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

# Pydantic models for request/response validation
class Features(BaseModel):
    features: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "qty_dot_url": 2.0,
                    "length_url": 45.0,
                    "qty_hyphen_url": 1.0
                }
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

class BatchRequest(BaseModel):
    samples: List[Features]

class BatchResponse(BaseModel):
    results: List[PredictionResponse]

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
scaler = None
input_dim = None

# Load model and scaler on startup
@app.on_event("startup")
async def load_model():
    global model, scaler, input_dim
    
    try:
        # Check if files exist
        scaler_path = 'scaler.pkl'
        model_path = 'best_phishing_model.pth'
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
        
        input_dim = len(scaler.feature_names_in_)
        
        # Load model
        model = PhishingNet(input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Device: {device}")
        print(f"✓ Input features: {input_dim}")
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained your model and saved 'best_phishing_model.pth'")
        print("2. Saved your scaler using: joblib.dump(scaler, 'scaler.pkl')")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def clean_qty_dot_url(val):
    """Clean qty_dot_url field"""
    if isinstance(val, str):
        nums = [float(x) for x in val.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
        return np.mean(nums) if nums else 0.0
    return float(val)

def preprocess_features(features: Dict[str, float]) -> torch.Tensor:
    """Preprocess input features for model inference"""
    # Clean qty_dot_url if present
    if 'qty_dot_url' in features:
        features['qty_dot_url'] = clean_qty_dot_url(features['qty_dot_url'])
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    expected_features = scaler.feature_names_in_
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
    
    # Reorder columns to match training
    df = df[expected_features]
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    return X_tensor

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Phishing Detection API",
        "status": "running",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "features": "/features",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "loaded" if model is not None else "not loaded",
        "device": str(device),
        "input_features": input_dim
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: Features):
    """
    Predict if URL features indicate phishing
    
    Returns:
    - prediction: "phishing" or "legitimate"
    - confidence: confidence percentage
    - probabilities: probability for each class
    """
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess features
        X_tensor = preprocess_features(data.features)
        
        # Make prediction
        with torch.no_grad():
            output = model(X_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return PredictionResponse(
            prediction='phishing' if predicted_class == 1 else 'legitimate',
            confidence=round(confidence * 100, 2),
            probabilities={
                'legitimate': round(probabilities[0][0].item() * 100, 2),
                'phishing': round(probabilities[0][1].item() * 100, 2)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchResponse)
async def batch_predict(data: BatchRequest):
    """
    Batch prediction for multiple samples
    
    Returns:
    - results: list of predictions for each sample
    """
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for sample in data.samples:
            # Preprocess features
            X_tensor = preprocess_features(sample.features)
            
            # Make prediction
            with torch.no_grad():
                output = model(X_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            results.append(
                PredictionResponse(
                    prediction='phishing' if predicted_class == 1 else 'legitimate',
                    confidence=round(confidence * 100, 2),
                    probabilities={
                        'legitimate': round(probabilities[0][0].item() * 100, 2),
                        'phishing': round(probabilities[0][1].item() * 100, 2)
                    }
                )
            )
        
        return BatchResponse(results=results)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/features")
def get_features():
    """Get list of expected input features"""
    try:
        if scaler is None:
            raise HTTPException(status_code=503, detail="Scaler not loaded")
        
        return {
            "feature_count": input_dim,
            "features": scaler.feature_names_in_.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 for local development
    uvicorn.run(app, host="127.0.0.1", port=8000)

