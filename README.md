# 🌱 Krishi_Kaar — Smart Agriculture & Environmental Monitoring System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Knight6azer/Krishi_kaar/graphs/commit-activity)

Precision agriculture is an emerging field that leverages technology to optimize crop yields and resource usage. This project presents a low-cost, IoT-based "Smart Agriculture & Environmental Monitoring System" designed to assist farmers in real-time decision-making. The system monitors critical soil and environmental parameters such as **soil moisture, air temperature, humidity, and water quality (TDS)** using an **Arduino Uno** controller, along with an **Ultrasonic Sensor** for imposter detection. Furthermore, it integrates **Artificial Intelligence (AI)** for crop health analysis, utilizing a **MobileNetV2 Convolutional Neural Network (CNN)** to detect plant diseases from images. It also features a **Recommendation Engine** that suggests optimal crops, fertilizers, water supply, and estimates market price. All data is visualized on an extensively translated multilingual web-based dashboard (supporting English and 7 Indian languages: **Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, and Urdu**), accompanied by dynamic UI elements like a "No Plant Detected" popup alert. An integrated alert system concurrently notifies the user of critical conditions. This prototype demonstrates the feasibility of combining IoT and ML to reduce agricultural losses and improve sustainability.

## 🚀 Features

- **Real-time Monitoring**: Soil Moisture, Air Temp/Humidity, and Water TDS
- **Crop Disease Detection**: MobileNetV2 CNN classifies camera frames as **Healthy / Diseased / No Plant**
- **Presence Detection**: Second MobileNetV2 model classifies frames as **Crop / Human / Imposter** — triggers camera-based imposter alerts
- **Water Quality Analysis**: Decision Tree Classifier predicts **Safe / Moderate / Unsafe**
- **Recommendation Engine**: Suggests Crop, Fertilizer, Water Supply, and Market Price based on NPK/pH sensor data
- **Multilingual UI**: Real-time translation supporting English and 7 Indian languages (Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu)
- **Premium Aesthetics**: Centered pill-shaped header with a modern glassmorphism effect
- **Streamlined View**: Hide-on-demand "Live Monitoring & Alerts" section to focus on core data
- **Dual-Layer Alerts**: Integrated camera-based ML vision and hardware Ultrasonic sensors
- **Intelligent Recommendations**: AI-driven guidance for Crop, Fertilizer, and Water Needs
- **Live Feed System**: Seamless MJPEG video streaming for real-time visual surveillance

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Hardware | Arduino Uno, DHT11, Soil & TDS Sensors, Ultrasonic, Webcam |
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
    └── dashboard.html       # Multilingual Web Dashboard (8 languages supported: EN + 7 Indian)
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
