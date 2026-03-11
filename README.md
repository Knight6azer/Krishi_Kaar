# 🌱 Krishi_Kaar — Smart Agriculture & Environmental Monitoring System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Knight6azer/Krishi_kaar/graphs/commit-activity)

A low-cost, IoT-based precision farming solution that integrates **Machine Learning** for crop disease detection, presence monitoring, and water quality assessment. Leverages Raspberry Pi 4 to monitor environmental parameters and provide actionable insights via a clean, multilingual web dashboard.

## 🚀 Features

- **Real-time Monitoring**: Soil Moisture, Air Temp/Humidity, Water TDS & Temperature
- **Crop Disease Detection**: MobileNetV2 CNN classifies camera frames as **Healthy / Diseased / No Plant**
- **Presence Detection**: Second MobileNetV2 model classifies frames as **Crop / Human / Imposter** — triggers camera-based imposter alerts
- **Water Quality Analysis**: Decision Tree Classifier predicts **Safe / Moderate / Unsafe**
- **Recommendation Engine**: Suggests Crop, Fertilizer, Water Supply, and Market Price based on NPK/pH sensor data
- **Multilingual UI**: Real-time translation across 8 languages (English + 7 Indian languages)
- **Alert System**: Dual-layer imposter detection via camera (ML) and Ultrasonic sensor
- **Toast Notifications**: "No Plant Detected" popup when camera frame is empty
- **Clean Light Theme**: White cards, green accents — readable and professional

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Hardware | Raspberry Pi 4, MCP3008, DHT11, DS18B20, Soil & TDS Sensors, Ultrasonic, Webcam |
| Software | Python 3, Flask, OpenCV, TensorFlow/Keras, Scikit-learn, JavaScript |

## 📂 Project Structure

```
Krishi_Kaar/
│
├── main.py                  # Main Flask Application
├── sensors.py               # Hardware Interface (Mock Mode for PC)
├── crop_cnn.py              # Crop Disease Detection (Healthy / Diseased / No Plant)
├── presence_classifier.py   # Presence Classifier (Crop / Human / Imposter)
├── water_ml.py              # Water Quality Classification
├── generate_graphs.py       # Report Graph Generator
├── requirements.txt         # Python Dependencies
└── templates/
    └── dashboard.html       # Multilingual Web Dashboard (8 languages)
```

## ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Knight6azer/Krishi_kaar.git
   cd Krishi_kaar
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```
   *The system auto-detects PC mode and switches to Mock sensor data.*

4. **Access the Dashboard**:
   Open [http://localhost:5000](http://localhost:5000) in your browser.

5. **Generate Report Graphs** (Optional):
   ```bash
   python generate_graphs.py
   ```

## 📊 Results

| Model | Accuracy |
|---|---|
| Crop Disease CNN (Healthy/Diseased/No Plant) | ~92% |
| Presence Classifier (Crop/Human/Imposter) | ~90% |
| Water Quality Decision Tree | ~95% |

- Sensor update latency: < 200ms
- Camera inference runs every 3 seconds (CPU-efficient)

## 📜 License

MIT License — Designed for Mini Project (S6), 2025.
