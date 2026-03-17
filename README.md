# Krishi_kaar: Industrial Smart Agriculture Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stack](https://img.shields.io/badge/Stack-Fullstack--AI-blue.svg)]()

Krishi_kaar is an industry-grade, startup-ready Smart Agriculture system. It leverages **IoT (Arduino Uno)**, **MongoDB**, and **Random Forest Machine Learning** to provide automated, data-driven farming assistance. 

## 🚀 Key Features

- **Industrial Dark Dashboard**: Professional UI with real-time analytics and Chart.js integration.
- **Dual Operational Modes**: 
    - **Smart Mode**: AI-driven automatic irrigation and pump control.
    - **Manual Mode**: Direct user control with real-time override.
- **Advanced Machine Learning**: Random Forest classification for irrigation, fertilization, and crop selection.
- **Data Persistence**: MongoDB integration for historical sensor tracking and configuration storage.
- **Farmer's Assistant**: Step-by-step guidance system for non-farmers based on AI predictions.
- **Multilingual Support**: Fully localized interface for regional accessibility.

## 🛠️ Technical Stack

| Layer | Technologies |
|---|---|
| **Hardware** | Arduino Uno, Soil Moisture, TDS, DHT11, Salinity (EC), Ultrasonic, Webcam |
| **Backend** | Python 3, Flask, MongoDB, PyMongo |
| **AI/ML** | Random Forest, CNN (MobileNetV2), Scikit-learn, TensorFlow |
| **Frontend** | Vanilla JS, Chart.js, Font Awesome, Glassmorphism CSS |

## 📂 Architecture

```bash
├── main.py               # Central Controller (Industrial Flask Server)
├── agri_ai.py            # Random Forest Inference Engine
├── sensors.py            # Arduino Hardware Interface
├── crop_cnn.py           # MobileNetV2 Vision Logic
├── templates/
│   └── dashboard.html    # Professional Dark Dashboard
└── static/               # High-fidelity assets
```

## ⚙️ Installation

1. **Prerequisites**: Python 3.9+, MongoDB Compass.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Server**:
   ```bash
   python main.py
   ```

---
*Developed for the next generation of precision agriculture.*
