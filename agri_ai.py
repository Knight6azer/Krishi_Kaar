import numpy as np
import pickle
import os
import random
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = 'agri_ai_model.pkl'

def train_agri_models():
    """
    Trains industrial-grade Random Forest models for:
    1. Irrigation Needs (Binary)
    2. Fertilizer Requirements (Multi-class)
    3. Crop Suitability (Multi-class)
    """
    X = []
    y_irrigation = []
    y_fertilizer = []
    y_crop = []
    
    # Dataset generation logic
    for _ in range(1500):
        # Features: [N, P, K, Temp, Humidity, pH, Moisture, Salinity]
        n = random.uniform(10, 200)
        p = random.uniform(5, 150)
        k = random.uniform(5, 150)
        temp = random.uniform(15, 45)
        hum = random.uniform(20, 95)
        ph = random.uniform(4, 9.5)
        moist = random.uniform(0, 100)
        sal = random.uniform(0.1, 8.0)
        
        X.append([n, p, k, temp, hum, ph, moist, sal])
        
        # 1. Irrigation Logic (Target: 0=OFF, 1=ON)
        # Based on moisture threshold and humidity
        if moist < 30 or (moist < 45 and temp > 35):
            y_irrigation.append(1)
        elif moist > 75:
            y_irrigation.append(0)
        else:
            y_irrigation.append(1 if random.random() > 0.6 else 0)
            
        # 2. Fertilizer Logic (0: None, 1: NPK, 2: Urea, 3: Potash, 4: DAP)
        if sal > 6.0: # High salinity - use specific treatment/none
            y_fertilizer.append(0) 
        elif n < 40 and p < 40 and k < 40:
            y_fertilizer.append(1)
        elif n < 35:
            y_fertilizer.append(2)
        elif k < 35:
            y_fertilizer.append(3)
        elif p < 35:
            y_fertilizer.append(4)
        else:
            y_fertilizer.append(0)
            
        # 3. Crop Suitability (0: Wheat, 1: Rice, 2: Potato, 3: Tomato, 4: Maize)
        if n > 110 and hum > 75:
            y_crop.append(1) # Rice
        elif temp > 32 and moist < 35:
            y_crop.append(4) # Maize
        elif ph < 6.0:
            y_crop.append(2) # Potato
        elif k > 90 and temp < 30:
            y_crop.append(3) # Tomato
        else:
            y_crop.append(0) # Wheat

    X = np.array(X)
    
    models = {
        'irrigation': RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_irrigation),
        'fertilizer': RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_fertilizer),
        'crop': RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_crop)
    }
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)
    print(f"Industrial Agriculture Models stored at {MODEL_FILE}")

def get_recommendations(readings):
    if not os.path.exists(MODEL_FILE):
        train_agri_models()
        
    try:
        with open(MODEL_FILE, 'rb') as f:
            models = pickle.load(f)
            
        input_data = np.array([[
            readings.get('nitrogen', 50),
            readings.get('phosphorus', 50),
            readings.get('potassium', 50),
            readings.get('air_temperature', 25),
            readings.get('humidity', 60),
            readings.get('ph', 7.0),
            readings.get('soil_moisture', 50),
            readings.get('salinity', 1.0)
        ]])
        
        # Predictions
        irr_pred = models['irrigation'].predict(input_data)[0]
        fert_pred = models['fertilizer'].predict(input_data)[0]
        
        # Top 3 Crops based on probabilities
        crop_probs = models['crop'].predict_proba(input_data)[0]
        crop_indices = np.argsort(crop_probs)[-3:][::-1]
        
        crop_labels = {0: "Wheat", 1: "Rice", 2: "Potato", 3: "Tomato", 4: "Maize"}
        top_crops = [crop_labels.get(idx, "Unknown") for idx in crop_indices]
        
        fert_labels = {0: "Safe Balance", 1: "Apply NPK 19:19:19", 2: "Apply Urea", 3: "Apply Potash", 4: "Apply DAP"}
        irr_labels = {0: "OFF", 1: "ON"}
        
        # Soil Health Score Calculation (Simplified logic)
        # Ideal: pH 6-7, Salinity < 2, Moisture 40-70
        ph = readings.get('ph', 7.0)
        sal = readings.get('salinity', 1.0)
        moist = readings.get('soil_moisture', 50)
        
        score = 100
        score -= abs(ph - 6.5) * 10
        score -= max(0, sal - 2) * 15
        if moist < 20 or moist > 80: score -= 20
        health_score = max(5, min(100, int(score)))

        return {
            "top_crops": top_crops,
            "crop": top_crops[0], # Primary one
            "fertilizer": fert_labels.get(fert_pred, "Stable"),
            "irrigation": irr_labels.get(irr_pred, "OFF"),
            "irrigation_code": int(irr_pred),
            "health_score": health_score
        }
    except Exception as e:
        print(f"Inference error: {e}")
        return {"top_crops": ["N/A"], "crop": "N/A", "fertilizer": "Stable", "irrigation": "OFF", "irrigation_code": 0, "health_score": 50}

if __name__ == "__main__":
    train_agri_models()
