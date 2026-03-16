import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier

MODEL_FILE = 'water_model.pkl'

def train_water_model():
    # Synthetic dataset for training
    # Features: [TDS (ppm), Temperature (C)]
    # Target: 0 (Safe), 1 (Moderate), 2 (Unsafe)
    
    X = np.array([
        [150, 25], [200, 24], [300, 26], [50, 22], [400, 28],  # Safe
        [600, 25], [700, 30], [550, 28], [800, 27], [650, 26], # Moderate
        [1200, 35], [1500, 32], [2000, 30], [1300, 34], [1100, 31], # Unsafe (High TDS)
        [300, 45], [250, 50], [200, 42], # Unsafe (High Temp)
    ])
    
    y = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        2, 2, 2
    ])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Water model trained and saved to {MODEL_FILE}")

def predict_water_quality(tds, temp=25.0):
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Training new model...")
        train_water_model()
        
    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)
        
    prediction = clf.predict([[tds, temp]])[0]
    labels = {0: "Safe", 1: "Moderate", 2: "Unsafe"}
    return labels.get(prediction, "Unknown")

if __name__ == "__main__":
    train_water_model()
    # Test
    print("Test Prediction (300 TDS, 25C):", predict_water_quality(300, 25))
    print("Test Prediction (1500 TDS, 30C):", predict_water_quality(1500, 30))
