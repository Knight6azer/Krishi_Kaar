"""
Krishi_Kaar — Water Quality ML Module (Production-Grade)

Predicts water quality (Safe / Moderate / Unsafe) from TDS and Temperature.
Thread-safe, cached model loading, expanded training data.
"""
import numpy as np
import pickle
import os
import threading
from sklearn.tree import DecisionTreeClassifier

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'water_model.pkl')

_model = None
_model_lock = threading.Lock()


def train_water_model():
    """Train water quality classifier with expanded realistic dataset."""
    # Features: [TDS (ppm), Temperature (°C)]
    # Target: 0=Safe, 1=Moderate, 2=Unsafe
    X = np.array([
        # Safe (low TDS, normal temp)
        [100, 22], [150, 25], [200, 24], [250, 26], [300, 23],
        [50, 20], [120, 27], [180, 25], [350, 24], [400, 26],
        [80, 23], [220, 25], [160, 21], [280, 24], [330, 27],
        # Moderate (medium TDS)
        [500, 25], [600, 26], [700, 28], [550, 27], [650, 29],
        [800, 27], [750, 26], [580, 30], [480, 25], [850, 28],
        [520, 24], [620, 26], [680, 29], [720, 27], [780, 28],
        # Unsafe (high TDS or extreme temp)
        [1000, 30], [1200, 35], [1500, 32], [2000, 30], [1300, 34],
        [1100, 31], [1800, 28], [2200, 33], [1600, 29], [1400, 36],
        [300, 45], [250, 48], [200, 42], [350, 44], [400, 46],
        [150, 50], [500, 40], [600, 43], [2500, 25], [1900, 31],
    ])
    
    y = np.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    ])
    
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X, y)
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    print(f"[WATER_ML] Model trained and saved to {MODEL_FILE}")
    return clf


def _load_model():
    """Thread-safe model loading with caching."""
    global _model
    if _model is not None:
        return _model
    
    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model
        
        if not os.path.exists(MODEL_FILE):
            print("[WATER_ML] Model not found. Training fresh model...")
            _model = train_water_model()
        else:
            with open(MODEL_FILE, 'rb') as f:
                _model = pickle.load(f)
        return _model


LABELS = {0: "Safe", 1: "Moderate", 2: "Unsafe"}

def predict_water_quality(tds, temp=25.0):
    """Predict water quality. Returns label string."""
    try:
        clf = _load_model()
        prediction = clf.predict([[float(tds), float(temp)]])[0]
        return LABELS.get(prediction, "Unknown")
    except Exception as e:
        print(f"[WATER_ML] Prediction error: {e}")
        return "Unknown"


if __name__ == "__main__":
    train_water_model()
    print("Test (300 TDS, 25°C):", predict_water_quality(300, 25))
    print("Test (800 TDS, 28°C):", predict_water_quality(800, 28))
    print("Test (1500 TDS, 30°C):", predict_water_quality(1500, 30))
    print("Test (200 TDS, 45°C):", predict_water_quality(200, 45))
