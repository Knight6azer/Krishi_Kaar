# 🌱 Krishi_Kaar — Smart Agriculture Platform

[![Platform](https://img.shields.io/badge/Platform-Agriculture--IoT-10b981.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9+-3b82f6.svg)]()
[![License](https://img.shields.io/badge/License-MIT-8b5cf6.svg)]()

**Krishi_Kaar** is a production-grade precision agriculture platform that combines real-time IoT sensor monitoring, machine learning predictions, and computer vision into a unified web dashboard. Designed for Indian farmers, it supports 8 regional languages and graceful degradation when hardware is unavailable.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Flask Server                      │
│  main.py — Routes, Auth, WebSocket, Video Stream    │
├──────────┬──────────┬──────────┬────────────────────┤
│ sensors  │ agri_ai  │ water_ml │ vision_models      │
│ (IoT)    │ (RF/DT)  │ (DT)    │ (MobileNetV2)      │
├──────────┴──────────┴──────────┴────────────────────┤
│      MongoDB → SQLite → In-Memory (auto-fallback)   │
└─────────────────────────────────────────────────────┘
```

## ✨ Features

### 🌡️ Real-Time Sensor Monitoring
- Soil moisture, temperature, humidity, pH, NPK, TDS, salinity
- **Arduino integration**: Plug in an Arduino and set `ARDUINO_PORT=COM5` 
- **Simulation mode**: Realistic temporally-coherent data when no hardware available

### 🧠 AI Intelligence
- **Crop Recommendation**: Random Forest trained on 2,200+ samples (22 crop classes)
- **Fertilizer Optimization**: ML-based recommendation from soil + climate data
- **Smart Irrigation**: Three modes — Manual, Rule-based, AI-driven
- **Water Quality**: TDS + temperature analysis (Safe/Moderate/Unsafe)
- **Soil Health Score**: Composite 0-100 score from pH, salinity, moisture, temperature

### 👁️ Computer Vision
- Crop disease detection (Healthy/Diseased/No Plant)
- Presence classification (Crop/Human/Imposter)
- Live camera feed streaming

### 🌍 Multilingual Support
8 Indian languages with localized digits:
English, हिंदी, मराठी, ગુજરાતી, தமிழ், తెలుగు, ಕನ್ನಡ, اردو

### 📊 Dashboard
- Real-time multi-sensor charts
- Interactive pump control
- Toast notification system for critical alerts
- Dark/Light theme
- Settings page for farm configuration
- Analytics page with historical trend analysis

---

## 📂 Project Structure

```
Krishi_Kaar/
├── main.py              # Flask server — API gateway & auth
├── config.py            # Centralized configuration
├── sensors.py           # Sensor module (Arduino + simulation)
├── agri_ai.py           # Random Forest crop/fertilizer/irrigation models
├── water_ml.py          # Water quality classifier
├── vision_models.py     # MobileNetV2 crop disease & presence detection
├── translations.py      # i18n dictionary (8 languages)
├── generate_graphs.py   # Static graph generation utility
├── templates/
│   ├── dashboard.html   # Main SaaS dashboard (JS + CSS)
│   └── auth.html        # Login/Signup page
├── static/              # Image assets
├── data/
│   ├── Crop_recommendation.csv
│   ├── Fertilizer Prediction.csv
│   └── sensor_log.csv
├── arduino/
│   └── led_control.ino  # Arduino firmware for sensor + LED control
├── requirements.txt     # Python dependencies
└── project_report.md    # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- (Optional) MongoDB for persistent storage
- (Optional) Arduino with soil/DHT sensors

### Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/Krishi_Kaar.git
cd Krishi_Kaar

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train AI models (first time only)
python agri_ai.py
python water_ml.py

# 4. Launch
python main.py
```

Open `http://localhost:5000` → Sign up → Access Dashboard.

### Environment Variables (Optional)

| Variable          | Default           | Description                        |
|-------------------|-------------------|------------------------------------|
| `KRISHI_SECRET_KEY` | auto-generated  | Flask session encryption key       |
| `MONGO_URI`       | `mongodb://localhost:27017/` | MongoDB connection string |
| `ARDUINO_PORT`    | None              | Serial port (e.g. `COM5`)          |
| `OWM_API_KEY`     | None              | OpenWeatherMap API key             |
| `WEATHER_CITY`    | `Bangalore`       | Default weather location           |
| `PORT`            | `5000`            | Server port                        |

---

## 🗄️ Database Hierarchy

The system gracefully degrades across three tiers:

1. **MongoDB** (if running) — Full persistence, user management, history
2. **SQLite** (auto-created) — Zero-config local persistence
3. **In-Memory** — Last resort, no persistence across restarts

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sensors` | GET | Latest sensor readings |
| `/api/recommendations` | GET | AI crop/fertilizer/irrigation predictions |
| `/api/vision` | GET | Camera vision status (disease + presence) |
| `/api/weather` | GET | Weather data (OWM or simulated) |
| `/api/history` | GET | Last 30 sensor readings |
| `/api/translations/<lang>` | GET | i18n dictionary for language code |
| `/api/control` | POST | Set mode or pump state |
| `/api/config` | POST | Update farm configuration |
| `/api/system_status` | GET | System health check |
| `/video_feed` | GET | MJPEG camera stream |

---

## 📡 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python, Flask, Flask-Login |
| **Database** | MongoDB / SQLite (auto-fallback) |
| **ML** | Scikit-learn (Random Forest), TensorFlow (MobileNetV2) |
| **Frontend** | Vanilla JS, Chart.js, CSS Glassmorphism |
| **IoT** | Arduino, pySerial |
| **i18n** | Custom engine with localized digit support |

---

*Built for the next generation of precision agriculture.*
