"""
Krishi_Kaar — Agriculture AI Engine (Production-Grade)

Trains and serves Random Forest models for:
1. Crop Recommendation (22 crops from N/P/K/Temp/Hum/pH/Rainfall)
2. Fertilizer Recommendation (from soil + climate features)
3. Irrigation Prediction (from moisture/temp/humidity)

Thread-safe, cached, with confidence scores and validation.
"""
import numpy as np
import pickle
import os
import threading
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agri_ai_model.pkl')

_models = None
_models_lock = threading.Lock()


def train_agri_models():
    """
    Train industrial-grade Random Forest models using real datasets.
    Returns the trained models dict.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    models = {}
    
    # --- 1. Crop Model ---
    crop_csv = os.path.join(data_dir, 'Crop_recommendation.csv')
    if os.path.exists(crop_csv):
        df_crop = pd.read_csv(crop_csv)
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X_crop = df_crop[feature_cols].values
        y_crop = df_crop['label'].values
        rf_crop = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_crop.fit(X_crop, y_crop)
        models['crop'] = rf_crop
        print(f"[AGRI_AI] Crop model trained on {len(X_crop)} samples, {len(rf_crop.classes_)} classes")
    else:
        print(f"[AGRI_AI] WARNING: {crop_csv} not found, using fallback")
        models['crop'] = RandomForestClassifier(n_estimators=10).fit([[50,50,50,25,60,7,50]], ["Unknown"])

    # --- 2. Fertilizer Model ---
    fert_csv = os.path.join(data_dir, 'Fertilizer Prediction.csv')
    if os.path.exists(fert_csv):
        df_fert = pd.read_csv(fert_csv)
        fert_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        X_fert = df_fert[fert_features].values
        y_fert = df_fert['Fertilizer Name'].values
        rf_fert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_fert.fit(X_fert, y_fert)
        models['fertilizer'] = rf_fert
        print(f"[AGRI_AI] Fertilizer model trained on {len(X_fert)} samples")
    else:
        print(f"[AGRI_AI] WARNING: {fert_csv} not found, using fallback")
        models['fertilizer'] = RandomForestClassifier(n_estimators=10).fit([[25,60,50,50,50,50]], ["Stable"])

    # --- 3. Irrigation Model ---
    sensor_csv = os.path.join(data_dir, 'sensor_log.csv')
    if os.path.exists(sensor_csv):
        df_sensor = pd.read_csv(sensor_csv)
        if 'moisture_percent' in df_sensor.columns and 'irrigation' in df_sensor.columns:
            X_irr = df_sensor[['moisture_percent', 'temperature', 'humidity']].values
            y_irr = df_sensor['irrigation'].values
            rf_irr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_irr.fit(X_irr, y_irr)
            models['irrigation'] = rf_irr
            print(f"[AGRI_AI] Irrigation model trained on {len(X_irr)} samples")
        else:
            print("[AGRI_AI] WARNING: sensor_log.csv missing expected columns")
            models['irrigation'] = RandomForestClassifier(n_estimators=10).fit([[50, 25, 60]], ["No Irrigation Needed"])
    else:
        print(f"[AGRI_AI] WARNING: {sensor_csv} not found, using fallback")
        models['irrigation'] = RandomForestClassifier(n_estimators=10).fit([[50, 25, 60]], ["No Irrigation Needed"])

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)
    print(f"[AGRI_AI] All models saved to {MODEL_FILE}")
    return models


def _load_models():
    """Thread-safe model loading with caching."""
    global _models
    if _models is not None:
        return _models
    
    with _models_lock:
        if _models is not None:
            return _models
        
        if not os.path.exists(MODEL_FILE):
            print("[AGRI_AI] Model file not found. Training fresh models...")
            _models = train_agri_models()
        else:
            try:
                with open(MODEL_FILE, 'rb') as f:
                    _models = pickle.load(f)
                # Validate model structure
                required_keys = ['crop', 'fertilizer', 'irrigation']
                if not all(k in _models for k in required_keys):
                    print("[AGRI_AI] Model file incomplete. Retraining...")
                    _models = train_agri_models()
            except Exception as e:
                print(f"[AGRI_AI] Model load error: {e}. Retraining...")
                _models = train_agri_models()
        return _models


def get_recommendations(readings):
    """
    Generate all AI recommendations from sensor readings.
    Thread-safe, returns dict with crops, fertilizer, irrigation, health score.
    """
    try:
        models = _load_models()
        
        n = float(readings.get('nitrogen', 50))
        p = float(readings.get('phosphorus', 50))
        k = float(readings.get('potassium', 50))
        t = float(readings.get('air_temperature', 25))
        h = float(readings.get('humidity', 60))
        ph_val = float(readings.get('ph', 7.0))
        m = float(readings.get('soil_moisture', 50))
        sal = float(readings.get('salinity', 1.0))
        
        # 1. Irrigation Prediction
        irr_pred = models['irrigation'].predict([[m, t, h]])[0]
        irrigation_on = 1 if "Required" in str(irr_pred) else 0
        irr_status = "ON" if irrigation_on else "OFF"
        
        # 2. Fertilizer Prediction
        fert_pred = str(models['fertilizer'].predict([[t, h, m, n, k, p]])[0])
        
        # 3. Crop Recommendation with confidence
        crop_probs = models['crop'].predict_proba([[n, p, k, t, h, ph_val, m]])[0]
        top_indices = np.argsort(crop_probs)[-3:][::-1]
        top_crops = []
        for idx in top_indices:
            crop_name = models['crop'].classes_[idx]
            confidence = round(float(crop_probs[idx]) * 100, 1)
            top_crops.append({"name": str(crop_name), "confidence": confidence})
        
        # Extract just names for backward compatibility
        top_crop_names = [c["name"] for c in top_crops]
        
        # 4. Health Score (0-100)
        score = 100.0
        score -= abs(ph_val - 6.5) * 8
        score -= max(0, sal - 2) * 12
        if m < 20: score -= (20 - m)
        if m > 80: score -= (m - 80)
        if t > 40: score -= (t - 40) * 3
        if t < 10: score -= (10 - t) * 3
        health_score = max(5, min(100, int(score)))

        return {
            "top_crops": top_crop_names,
            "top_crops_detailed": top_crops,
            "crop": top_crop_names[0] if top_crop_names else "Unknown",
            "fertilizer": fert_pred,
            "irrigation": irr_status,
            "irrigation_code": irrigation_on,
            "irrigation_label": str(irr_pred),
            "health_score": health_score
        }
    except Exception as e:
        print(f"[AGRI_AI] Inference error: {e}")
        return {
            "top_crops": ["N/A"], 
            "top_crops_detailed": [{"name": "N/A", "confidence": 0}],
            "crop": "N/A", 
            "fertilizer": "Stable", 
            "irrigation": "OFF", 
            "irrigation_code": 0, 
            "irrigation_label": "Error",
            "health_score": 50
        }


if __name__ == "__main__":
    print("[AGRI_AI] Training models from scratch...")
    train_agri_models()
    
    # Test inference
    test_readings = {
        "nitrogen": 90, "phosphorus": 42, "potassium": 43,
        "air_temperature": 25, "humidity": 80, "ph": 6.5,
        "soil_moisture": 45, "salinity": 1.2
    }
    result = get_recommendations(test_readings)
    print(f"\nTest Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
