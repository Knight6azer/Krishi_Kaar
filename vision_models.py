import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os
import random

IMG_SIZE = 224

# --- Frame Quality Analysis ---
# These thresholds can be overridden from config at runtime
_VARIANCE_MIN = 15.0
_BRIGHTNESS_MIN = 10
_BRIGHTNESS_MAX = 245
_CONFIDENCE_THRESHOLD = 0.55

try:
    from config import Config
    _VARIANCE_MIN = Config.VISION_FRAME_VARIANCE_MIN
    _BRIGHTNESS_MIN = Config.VISION_BRIGHTNESS_MIN
    _BRIGHTNESS_MAX = Config.VISION_BRIGHTNESS_MAX
    _CONFIDENCE_THRESHOLD = Config.VISION_CONFIDENCE_THRESHOLD
except Exception:
    pass


def analyze_frame_quality(frame):
    """
    Analyze camera frame quality to detect blocked cameras, darkness, overexposure.
    Returns: dict with 'valid' bool, 'issue' string, and quality metrics.
    """
    if frame is None:
        return {"valid": False, "issue": "No Frame", "brightness": 0, "variance": 0, "edges": 0}

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Mean brightness
        brightness = float(np.mean(gray))

        # Pixel variance — low means uniform/blocked
        variance = float(np.std(gray))

        # Edge density — no edges means no real content
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size * 100  # percentage of edge pixels

        # Decision logic
        if variance < _VARIANCE_MIN:
            if brightness < _BRIGHTNESS_MIN:
                return {"valid": False, "issue": "Camera Blocked", "brightness": brightness, "variance": variance, "edges": edge_density}
            elif brightness > _BRIGHTNESS_MAX:
                return {"valid": False, "issue": "Overexposed", "brightness": brightness, "variance": variance, "edges": edge_density}
            else:
                return {"valid": False, "issue": "No Content", "brightness": brightness, "variance": variance, "edges": edge_density}

        if brightness < _BRIGHTNESS_MIN:
            return {"valid": False, "issue": "Too Dark", "brightness": brightness, "variance": variance, "edges": edge_density}

        if brightness > _BRIGHTNESS_MAX:
            return {"valid": False, "issue": "Overexposed", "brightness": brightness, "variance": variance, "edges": edge_density}

        if edge_density < 0.5:
            return {"valid": False, "issue": "No Content", "brightness": brightness, "variance": variance, "edges": edge_density}

        return {"valid": True, "issue": None, "brightness": brightness, "variance": variance, "edges": edge_density}

    except Exception as e:
        return {"valid": False, "issue": f"Analysis Error", "brightness": 0, "variance": 0, "edges": 0}


def build_generic_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class GenericVisionClassifier:
    def __init__(self, model_file, num_classes, labels):
        self.model_file = model_file
        self.num_classes = num_classes
        self.labels = labels
        self.model = None

    def _ensure_model(self):
        if not os.path.exists(self.model_file):
            print(f"Model file {self.model_file} not found. Creating a fresh (untrained) model for testing pipeline...")
            model = build_generic_model(self.num_classes)
            model.save(self.model_file)

    def train_model(self, train_dir, val_dir):
        model = build_generic_model(self.num_classes)
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            # Implementation details for actual training
            pass
        else:
            print(f"Training directories not found. Saving an untrained model architecture to {self.model_file} for testing.")
            model.save(self.model_file)

    def predict(self, image_path_or_array):
        """
        Run prediction with frame quality validation and confidence thresholding.
        Returns dict with: label, confidence, quality (issue info), raw_label, raw_confidence
        """
        self._ensure_model()
        
        # Resolve image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array

        if image is None:
            return {"label": "No Signal", "confidence": 0.0, "quality": "No Frame"}

        # --- Step 1: Frame quality check ---
        quality = analyze_frame_quality(image)
        if not quality["valid"]:
            return {
                "label": quality["issue"],
                "confidence": 0.0,
                "quality": quality["issue"],
                "quality_details": quality
            }

        # --- Step 2: Model prediction ---
        try:
            if self.model is None:
                self.model = tf.keras.models.load_model(self.model_file)

            img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            
            preds = self.model.predict(img, verbose=0)
            label_idx = np.argmax(preds)
            confidence = float(np.max(preds))
            raw_label = self.labels[label_idx]
            
            # --- Step 3: Confidence thresholding ---
            if confidence < _CONFIDENCE_THRESHOLD:
                return {
                    "label": "Uncertain",
                    "confidence": confidence,
                    "quality": "Low Confidence",
                    "raw_label": raw_label,
                    "raw_confidence": confidence
                }
            
            return {
                "label": raw_label,
                "confidence": confidence,
                "quality": "Good"
            }
            
        except Exception as e:
            print(f"Prediction Error ({self.model_file}): {e}")
            return {"label": "Model Error", "confidence": 0.0, "quality": "Error"}


# Initialize the modular classifiers based on constants from original files
crop_classifier = GenericVisionClassifier(
    model_file='crop_cnn_model.h5',
    num_classes=3,
    labels={0: "Healthy", 1: "Diseased", 2: "No Plant"}
)

presence_classifier_instance = GenericVisionClassifier(
    model_file='presence_cnn_model.h5',
    num_classes=3,
    labels={0: "Crop", 1: "Human", 2: "Imposter"}
)

# Exposed methods for the main app
def predict_crop_disease(image):
    return crop_classifier.predict(image)

def predict_presence(image):
    return presence_classifier_instance.predict(image)

if __name__ == "__main__":
    if not os.path.exists(crop_classifier.model_file):
        crop_classifier.train_model('dummy_train', 'dummy_val')
    if not os.path.exists(presence_classifier_instance.model_file):
         presence_classifier_instance.train_model('dummy_train', 'dummy_val')
    
    # Test with blocked frame (all black)
    black_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    print("Black frame (blocked camera):", predict_crop_disease(black_frame))
    print("Black frame presence:", predict_presence(black_frame))

    # Test with real-ish frame (random noise — has variance)
    noise_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    print("Noise frame crop:", predict_crop_disease(noise_frame))
    print("Noise frame presence:", predict_presence(noise_frame))
