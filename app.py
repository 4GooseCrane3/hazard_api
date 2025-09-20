from fastapi import FastAPI, Request
import torch
import tensorflow as tf
import joblib
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# ------------------------------
# Load models
# ------------------------------

# 1️⃣ PyTorch / Ultralytics model
try:
    pt_model = YOLO("ripcurrent_model.pt")  # loads YOLO model safely
except Exception as e:
    print("Error loading YOLO model:", e)
    pt_model = None

# 2️⃣ Keras model
try:
    keras_model = tf.keras.models.load_model("hightide_model.keras")
except Exception as e:
    print("Error loading Keras model:", e)
    keras_model = None

# 3️⃣ Scikit-learn model
try:
    joblib_model = joblib.load("flood_model.joblib")
except Exception as e:
    print("Error loading joblib model:", e)
    joblib_model = None

# ------------------------------
# API endpoint
# ------------------------------
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    alert_type = data.get("alert_type")
    features = np.array(data.get("features", []), dtype=np.float32)

    if alert_type == "rip_current":
        if pt_model is None:
            return {"error": "PyTorch model not loaded"}
        pred = pt_model.predict(features)  # adapt if using images
        severity = float(pred[0].boxes.conf[0]) if pred[0].boxes else 0.0
    elif alert_type == "high_tide":
        if keras_model is None:
            return {"error": "Keras model not loaded"}
        features_keras = features.reshape(1, -1)
        pred = keras_model.predict(features_keras)
        severity = float(pred[0][0])
    elif alert_type == "flood":
        if joblib_model is None:
            return {"error": "Joblib model not loaded"}
        features_joblib = features.reshape(1, -1)
        pred = joblib_model.predict(features_joblib)
        severity = float(pred[0])
    else:
        return {"error": "Unknown alert type"}

    return {"alert_type": alert_type, "severity": severity}
