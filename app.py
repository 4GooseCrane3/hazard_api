from fastapi import FastAPI, Request
import numpy as np
import joblib
import tensorflow as tf
from ultralytics import YOLO
import os
import uvicorn

app = FastAPI()

# ------------------------------
# Root endpoint (Render health check)
# ------------------------------
@app.get("/")
async def root():
    return {"message": "Hazard API is running"}

# ------------------------------
# Lazy model loaders
# ------------------------------
_pt_model = None
_keras_model = None
_joblib_model = None

def get_pt_model():
    global _pt_model
    if _pt_model is None:
        try:
            _pt_model = YOLO("ripcurrent_model.pt")
        except Exception as e:
            print("YOLO load error:", e)
            _pt_model = None
    return _pt_model

def get_keras_model():
    global _keras_model
    if _keras_model is None:
        try:
            _keras_model = tf.keras.models.load_model("hightide_model.keras")
        except Exception as e:
            print("Keras load error:", e)
            _keras_model = None
    return _keras_model

def get_joblib_model():
    global _joblib_model
    if _joblib_model is None:
        try:
            _joblib_model = joblib.load("flood_model.joblib")
        except Exception as e:
            print("Joblib load error:", e)
            _joblib_model = None
    return _joblib_model

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    alert_type = data.get("alert_type")
    features = np.array(data.get("features", []), dtype=np.float32)

    if alert_type == "rip_current":
        model = get_pt_model()
        if model is None:
            return {"error": "PyTorch YOLO model not loaded"}
        # Placeholder: replace with actual YOLO inference
        return {"alert_type": alert_type, "severity": 1.0}

    elif alert_type == "high_tide":
        model = get_keras_model()
        if model is None:
            return {"error": "Keras model not loaded"}
        features_keras = features.reshape(1, -1)
        pred = model.predict(features_keras)
        return {"alert_type": alert_type, "severity": float(pred[0][0])}

    elif alert_type == "flood":
        model = get_joblib_model()
        if model is None:
            return {"error": "Joblib model not loaded"}
        features_joblib = features.reshape(1, -1)
        pred = model.predict(features_joblib)
        return {"alert_type": alert_type, "severity": float(pred[0])}

    else:
        return {"error": "Unknown alert type"}

# ------------------------------
# Run server (Render-ready)
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
