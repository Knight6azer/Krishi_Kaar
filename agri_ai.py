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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from config import Config
    MODEL_FILE = Config.AGRI_MODEL_FILE
    DATA_DIR = Config.DATA_DIR
except Exception:
    MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agri_ai_model.pkl')
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

_models = None
_models_lock = threading.Lock()


def train_agri_models():
    """
    Train industrial-grade Random Forest models using real datasets.
    Evaluates on holdout test sets and records performance metrics.
    Returns the trained models dict.
    """
    models = {}
    metrics = {}
    
    # --- 1. Crop Model ---
    crop_csv = os.path.join(DATA_DIR, 'Crop_recommendation.csv')
    if os.path.exists(crop_csv):
        df_crop = pd.read_csv(crop_csv)
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X_crop = df_crop[feature_cols].values
        y_crop = df_crop['label'].values
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42, stratify=y_crop)
        rf_crop = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_crop.fit(X_tr, y_tr)
        
        y_pred = rf_crop.predict(X_te)
        c_acc = accuracy_score(y_te, y_pred)
        c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(y_te, y_pred, average='weighted', zero_division=0)
        metrics['crop'] = {'accuracy': c_acc, 'precision': c_prec, 'recall': c_rec, 'f1': c_f1, 'samples': len(X_crop)}
        print(f"[AGRI_AI] Crop model test metrics — Acc: {c_acc:.4f}, Prec: {c_prec:.4f}, Rec: {c_rec:.4f}, F1: {c_f1:.4f}")
        
        # Train on full data for maximum operational coverage
        rf_crop.fit(X_crop, y_crop)
        models['crop'] = rf_crop
        models['crop_features'] = feature_cols
    else:
        print(f"[AGRI_AI] WARNING: {crop_csv} not found, using fallback")
        models['crop'] = RandomForestClassifier(n_estimators=10).fit([[50,50,50,25,60,7,50]], ["Unknown"])
        models['crop_features'] = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    # --- 2. Fertilizer Model ---
    fert_csv = os.path.join(DATA_DIR, 'Fertilizer Prediction.csv')
    if os.path.exists(fert_csv):
        df_fert = pd.read_csv(fert_csv)
        fert_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        X_fert = df_fert[fert_features].values
        y_fert = df_fert['Fertilizer Name'].values
        
        X_f_tr, X_f_te, y_f_tr, y_f_te = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42, stratify=y_fert)
        rf_fert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_fert.fit(X_f_tr, y_f_tr)
        
        y_f_pred = rf_fert.predict(X_f_te)
        f_acc = accuracy_score(y_f_te, y_f_pred)
        f_prec, f_rec, f_f1, _ = precision_recall_fscore_support(y_f_te, y_f_pred, average='weighted', zero_division=0)
        metrics['fertilizer'] = {'accuracy': f_acc, 'precision': f_prec, 'recall': f_rec, 'f1': f_f1, 'samples': len(X_fert)}
        print(f"[AGRI_AI] Fertilizer model test metrics — Acc: {f_acc:.4f}, Prec: {f_prec:.4f}, Rec: {f_rec:.4f}, F1: {f_f1:.4f}")
        
        rf_fert.fit(X_fert, y_fert)
        models['fertilizer'] = rf_fert
        models['fertilizer_features'] = fert_features
    else:
        print(f"[AGRI_AI] WARNING: {fert_csv} not found, using fallback")
        models['fertilizer'] = RandomForestClassifier(n_estimators=10).fit([[25,60,50,50,50,50]], ["Stable"])
        models['fertilizer_features'] = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

    # --- 3. Irrigation Model ---
    sensor_csv = os.path.join(DATA_DIR, 'sensor_log.csv')
    if os.path.exists(sensor_csv):
        df_sensor = pd.read_csv(sensor_csv)
        if 'moisture_percent' in df_sensor.columns and 'irrigation' in df_sensor.columns:
            X_irr = df_sensor[['moisture_percent', 'temperature', 'humidity']].values
            y_irr = df_sensor['irrigation'].values
            
            X_i_tr, X_i_te, y_i_tr, y_i_te = train_test_split(X_irr, y_irr, test_size=0.2, random_state=42, stratify=y_irr)
            rf_irr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_irr.fit(X_i_tr, y_i_tr)
            
            y_i_pred = rf_irr.predict(X_i_te)
            i_acc = accuracy_score(y_i_te, y_i_pred)
            i_prec, i_rec, i_f1, _ = precision_recall_fscore_support(y_i_te, y_i_pred, average='weighted', zero_division=0)
            metrics['irrigation'] = {'accuracy': i_acc, 'precision': i_prec, 'recall': i_rec, 'f1': i_f1, 'samples': len(X_irr)}
            print(f"[AGRI_AI] Irrigation model test metrics — Acc: {i_acc:.4f}, Prec: {i_prec:.4f}, Rec: {i_rec:.4f}, F1: {i_f1:.4f}")
            
            rf_irr.fit(X_irr, y_irr)
            models['irrigation'] = rf_irr
            models['irrigation_features'] = ['moisture_percent', 'temperature', 'humidity']
        else:
            print("[AGRI_AI] WARNING: sensor_log.csv missing expected columns")
            models['irrigation'] = RandomForestClassifier(n_estimators=10).fit([[50, 25, 60]], ["No Irrigation Needed"])
    else:
        print(f"[AGRI_AI] WARNING: {sensor_csv} not found, using fallback")
        models['irrigation'] = RandomForestClassifier(n_estimators=10).fit([[50, 25, 60]], ["No Irrigation Needed"])

    models['metrics'] = metrics

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)
    print(f"[AGRI_AI] All models and metrics saved to {MODEL_FILE}")
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
        
        # Safe input conversion with physical domain bounds clamping
        n = max(0.0, min(300.0, float(readings.get('nitrogen', 50) if readings.get('nitrogen') is not None else 50)))
        p = max(0.0, min(300.0, float(readings.get('phosphorus', 50) if readings.get('phosphorus') is not None else 50)))
        k = max(0.0, min(300.0, float(readings.get('potassium', 50) if readings.get('potassium') is not None else 50)))
        t = max(-10.0, min(65.0, float(readings.get('air_temperature', 25) if readings.get('air_temperature') is not None else 25)))
        h = max(0.0, min(100.0, float(readings.get('humidity', 60) if readings.get('humidity') is not None else 60)))
        ph_val = max(0.0, min(14.0, float(readings.get('ph', 7.0) if readings.get('ph') is not None else 7.0)))
        m = max(0.0, min(100.0, float(readings.get('soil_moisture', 50) if readings.get('soil_moisture') is not None else 50)))
        sal = max(0.0, min(20.0, float(readings.get('salinity', 1.0) if readings.get('salinity') is not None else 1.0)))
        
        # Rainfall: use explicit reading if provided; otherwise correlate with moisture
        if 'rainfall' in readings and readings['rainfall'] is not None:
            rainfall = max(0.0, min(500.0, float(readings['rainfall'])))
        else:
            # Scaled proxy where 50% moisture corresponds to standard ~100mm regional precipitation
            rainfall = max(20.0, min(300.0, m * 2.0))
        
        # 1. Irrigation Prediction
        irr_pred = models['irrigation'].predict([[m, t, h]])[0]
        irrigation_on = 1 if "Required" in str(irr_pred) else 0
        irr_status = "ON" if irrigation_on else "OFF"
        
        # 2. Fertilizer Prediction
        fert_pred = str(models['fertilizer'].predict([[t, h, m, n, k, p]])[0])
        
        # 3. Crop Recommendation with confidence
        crop_probs = models['crop'].predict_proba([[n, p, k, t, h, ph_val, rainfall]])[0]
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
