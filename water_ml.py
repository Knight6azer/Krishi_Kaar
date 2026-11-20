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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from config import Config
    MODEL_FILE = Config.WATER_MODEL_FILE
except Exception:
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    print(f"[WATER_ML] Holdout Test Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Retrain on full dataset for serving
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
    """Predict water quality with input validation. Returns label string."""
    try:
        tds_val = float(tds) if tds is not None else 350.0
        temp_val = float(temp) if temp is not None else 25.0
        # Realistic clamping
        tds_val = max(0.0, min(5000.0, tds_val))
        temp_val = max(-10.0, min(70.0, temp_val))
        
        clf = _load_model()
        prediction = clf.predict([[tds_val, temp_val]])[0]
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
