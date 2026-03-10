# SMART AGRICULTURE & ENVIRONMENTAL MONITORING SYSTEM
## Mini Project Report
Project Code: **MP(S6)-2025**

---

### 1. ABSTRACT
Precision agriculture is an emerging field that leverages technology to optimize crop yields and resource usage. This project presents a low-cost, IoT-based "Smart Agriculture & Environmental Monitoring System" designed to assist farmers in real-time decision-making. The system monitors critical soil and environmental parameters such as **soil moisture, air temperature, humidity, and water quality (TDS and Temperature)** using a Raspberry Pi 4 controller, along with an **Ultrasonic Sensor** for imposter detection. Furthermore, it integrates **Artificial Intelligence (AI)** for crop health analysis, utilizing a **MobileNetV2 Convolutional Neural Network (CNN)** to detect plant diseases from images. It also features a **Recommendation Engine** that suggests optimal crops, fertilizers, water supply, and estimates market price. All data is visualized on a multilingual web-based dashboard (supporting English and 7 Indian languages), and an alert system notifies the user of critical conditions. This prototype demonstrates the feasibility of combining IoT and ML to reduce agricultural losses and improve sustainability.

---

### 2. INTRODUCTION

#### 2.1 Background
Agriculture is the backbone of the economy, yet it faces challenges like unpredictable weather, water scarcity, and crop diseases. Traditional farming relies heavily on manual observation, which is labor-intensive and often inaccurate. The advent of the **Internet of Things (IoT)** allows for precise monitoring of field conditions, while **Machine Learning (ML)** offers predictive capabilities that can prevent crop failure.

#### 2.2 Problem Statement
Farmers often lack real-time data regarding their soil and water conditions, leading to over-irrigation or crop stress. Additionally, early detection of crop diseases is difficult for the naked eye until significant damage has occurred. Existing commercial solutions are expensive and complex to deploy for small-scale farmers.

#### 2.3 Objectives
- To design an IoT sensing node for soil moisture, temperature, humidity, water quality, and ultrasonic distance.
- To implement a Computer Vision model for identifying common crop diseases.
- To develop a Machine Learning model for classifying irrigation water quality.
- To build a Recommendation Engine providing insights on crop selection, fertilizers, and water needs.
- To create a unified, multilingual web dashboard for data visualization and remote monitoring.
- To implement an automated alert system for critical environmental thresholds and intruder detection.

---

### 3. SYSTEM ARCHITECTURE

#### 3.1 Overview
The system is built around a centralized controller (Raspberry Pi 4) which acts as the edge computing node. It interfaces with various analog and digital sensors to collect environmental data.
- **Input Layer**: Sensors (Soil Moisture, DHT11, TDS, DS18B20) and Camera.
- **Processing Layer**: Raspberry Pi reads sensor data, digitizes analog signals via MCP3008, and performs ML inference.
- **Output Layer**: Flask Web Server (Dashboard) and Alert System (Email/Console).

#### 3.2 Hardware Components
1. **Raspberry Pi 4 Model B**: Selected for its high processing power (Quad-core Cortex-A72), 4GB RAM, and support for full Python ML libraries (TensorFlow, OpenCV).
2. **MCP3008 ADC**: An 8-channel, 10-bit Analog-to-Digital Converter used to interface the analog Soil Moisture and TDS sensors with the Pi's digital GPIO.
3. **Soil Moisture Sensor**: Capacitive/Resistive sensor measuring volumetric water content.
4. **DHT11**: A basic, low-cost digital temperature and humidity sensor.
5. **TDS Sensor (Total Dissolved Solids)**: Measures the conductivity of water to estimate purity.
6. **DS18B20**: A waterproof digital temperature sensor for water temperature monitoring.
7. **Ultrasonic Sensor**: Measures distance to detect potential imposters or animals entering the field.

---

### 4. SOFTWARE DESIGN & METHODOLOGY

#### 4.1 Technology Stack
- **Languages**: Python 3.7+ (Core logic), HTML/CSS/JavaScript (Frontend).
- **Frameworks**: Flask (Web Framework), TensorFlow/Keras (Deep Learning), Scikit-learn (ML).
- **Libraries**: OpenCV (Image Processing), RPi.GPIO (Hardware Control), Matplotlib (Analysis).

#### 4.2 Machine Learning Models

**A. Crop Disease Detection (CNN)**
- **Architecture**: **MobileNetV2** was selected for its balance between accuracy and computational efficiency, making it ideal for edge devices like the Raspberry Pi. It utilizes **inverted residual blocks** and **depth-wise separable convolutions** to minimize parameter count.
- **Training Strategy**: **Transfer Learning** was employed. The base layers, pre-trained on the massive ImageNet dataset, were frozen to retain feature extraction capabilities. A custom classification head (GlobalAveragePooling -> Dense 128 (ReLU) -> Dropout 0.5 -> Softmax 3) was added and trained on our specific dataset.
- **Dataset**: A curated subset of the **PlantVillage dataset**, comprising 3 classes: "Healthy", "Diseased" (e.g., Early Blight, Rust), and "Other" (background/irrelevant).
- **Performance**: The model achieves **~92% accuracy** on the validation set with an inference time of <100ms on the Pi 4.

**B. Water Quality Classification**
- **Algorithm**: Decision Tree Classifier.
- **Features**: TDS (Total Dissolved Solids) in ppm, Temperature in °C.
- **Target Classes**: Safe, Moderate, Unsafe.
- **Reasoning**: Decision Trees are non-parametric and fast, making them ideal for simple rule-based classification tasks based on tabular sensor data.

---

### 5. IMPLEMENTATION DETAILS

#### 5.1 Sensor Interfacing
- The **MCP3008** is connected via the SPI interface (SCLK, MOSI, MISO, CE0).
- **DHT11** uses a single GPIO pin with a proprietary one-wire protocol.
- **DS18B20** uses the 1-Wire protocol (GPIO 4 usually, configurable).

#### 5.2 Dashboard Development
The dashboard is a responsive web application served by Flask.
- **Backend API**: `/api/sensors` returns JSON data.
- **Frontend**: JavaScript `fetch()` polls the API every 2 seconds to update the DOM without reloading the page.
- **Video Stream**: A generator function yields JPEG frames from the camera to a `<img>` tag source, creating a seamless video feed.

---

### 6. RESULTS AND DISCUSSION

#### 6.1 Environmental Monitoring
The system successfully logged temperature and humidity data over a 7-day test period. The soil moisture sensor reacted instantly to watering events, showing a drop in resistance (increase in percentage).

#### 6.2 ML Inference
- **Vision**: The camera feed has a latency of ~200ms. Disease detection occurs every 3 seconds to preserve CPU resources.
- **Water Quality**: The Decision Tree model predicts "Safe" or "Unsafe" instantly (<1ms) upon reading the TDS sensor.

#### 6.3 Alerting
Simulating a "High TDS" event (salt water test) triggered a console alert and changed the dashboard status to "Unsafe" (Red) within 2 seconds.

---

### 7. CONCLUSION
The project successfully demonstrates a holistic approach to smart farming. By integrating low-cost sensors with powerful edge AI, we provide farmers with actionable insights that can save water and reduced crop loss. The system is modular, scalable, and user-friendly.

### 8. FUTURE SCOPE
- **Cloud Integration**: Uploading data to AWS/ThingsSpeak for long-term historical analysis.
- **Automation**: Connecting a relay module to automatically turn on the water pump when soil moisture is low.
- **Mobile App**: Developing a React Native app for push notifications.

---

### 9. REFERENCES
[1] "PlantVillage Dataset," arXiv:1511.08060, 2015.
[2] MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.
[3] Raspberry Pi 4 Datasheet, Raspberry Pi Foundation.
