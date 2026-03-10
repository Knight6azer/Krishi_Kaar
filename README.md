# 🌱 Smart Agriculture & Environmental Monitoring System
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Knight6azer/Krishi_kaar/graphs/commit-activity)

A low-cost, IoT-based precision farming solution that integrates **Machine Learning** for crop disease detection and water quality assessment. This project leverages Raspberry Pi 4 to monitor environmental parameters and provide actionable insights via a web dashboard.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Predictive+Project+Dashboard)

## 🚀 Features

- **Real-time Monitoring**: Soil Moisture, Air Temp/Humidity, Water TDS & Temp.
- **Crop Disease Detection**: Uses **MobileNetV2** (CNN) to identify plant diseases from camera feed.
- **Water Quality Analysis**: Uses **Decision Tree Classifier** to determine safe/unsafe water conditions.
- **Recommendation Engine**: Provides Crop, Fertilizer, Water, and Market Price recommendations based on sensing data.
- **Multilingual Web Dashboard**: Interactive Flask-based interface translated into 7 major Indian languages + English.
- **Alert System & Imposter Detection**: Notifications for critical conditions and an Ultrasonic sensor to detect intruders.

## 🛠️ Tech Stack

- **Hardware**: Raspberry Pi 4, MCP3008 ADC, DHT11, DS18B20, Soil Moisture Sensor, TDS Sensor, Ultrasonic Sensor, Webcam.
- **Software**: Python 3, Flask, OpenCV, TensorFlow/Keras, Scikit-learn.

## 📂 Project Structure

```
MP(S6)/
│
├── main.py              # Main Flask Application
├── sensors.py           # Hardware Interface (with Mock Mode)
├── crop_cnn.py          # Crop Disease Detection Model
├── water_ml.py          # Water Quality Classification Model
├── generate_graphs.py   # Utility to create report graphs
├── requirements.txt     # Dependencies
└── templates/
    └── dashboard.html   # Web Interface
```

## ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/smart-agriculture.git
   cd smart-agriculture
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```
   ```
   *The system will automatically detect if it is running on a PC and switch to **Mock Mode** for sensors.*

4. **Generate Report Graphs** (Optional):
   ```bash
   python generate_graphs.py
   ```
   This will create visualization images (`graph_*.png`) used in the project report.

5. **Access the Dashboard**:
   Open [http://localhost:5000](http://localhost:5000) in your browser.

## 📊 Results

- **CNN Accuracy**: ~92%
- **Water Quality Accuracy**: ~95%
- Latency: < 200ms sensor updates

## 📜 License

MIT License. Designed for Mini Project (S6).
