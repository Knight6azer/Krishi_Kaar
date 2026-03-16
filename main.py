"""
Krishi_Kaar & Environmental Monitoring System - Main Application
This Flask application serves as the central controller for the IoT system.
It aggregates data from sensors, runs ML inference for crop disease and water quality,
and serves a real-time dashboard.
"""
from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import sensors
import water_ml
import crop_cnn
import presence_classifier
import numpy as np

app = Flask(__name__)

# Global variables to store latest data
latest_sensor_data = {}
latest_crop_status = {"label": "Waiting...", "confidence": "0"}
latest_presence_status = {"label": "Waiting...", "confidence": "0"}
camera = None

def get_camera():
    global camera
    if camera is None:
        # Try to open camera (0 is usually the default webcam)
        camera = cv2.VideoCapture(0)
        # If unable to open, we might need a fallback or keep trying
    return camera

def generate_frames():
    global latest_crop_status
    cam = get_camera()
    while True:
        if cam is not None and cam.isOpened():
            success, frame = cam.read()
            if not success:
                break
            else:
                # Run crop disease detection on this frame occasionally (not every frame to save resources)
                # For demo, we can just save it to a global variable or run prediction logic here
                # Here we just encode it for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
             # Fallback if no camera (Mock black image)
             blank_image = np.zeros((480, 640, 3), np.uint8)
             cv2.putText(blank_image, "No Camera Found", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             ret, buffer = cv2.imencode('.jpg', blank_image)
             frame_bytes = buffer.tobytes()
             yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
             time.sleep(1)


# Alert Logic
def send_email_alert(subject, body):
    # Placeholder for email sending logic using smtplib
    # In a real scenario, you would configure SMTP server details here
    print(f"!!! ALERT !!! [{subject}] {body}")

def check_alerts(sensor_data, crop_data, presence_data):
    # 1. Soil Moisture Alert
    if isinstance(sensor_data.get('soil_moisture'), (int, float)) and sensor_data['soil_moisture'] < 30:
        send_email_alert("Low Soil Moisture", f"Soil Moisture is critical: {sensor_data['soil_moisture']}%")
        
    # 2. Water Quality Alert
    if sensor_data.get('water_quality') == 'Unsafe':
        send_email_alert("Unsafe Water Quality", f"Water is Unsafe! TDS: {sensor_data.get('tds')} ppm")
        
    # 3. Crop Disease Alert
    if crop_data.get('label') == 'Diseased':
        send_email_alert("Crop Disease Detected", f"Disease detected with confidence {crop_data.get('confidence')}%")
        
    # 4. Imposter Alert (Camera & Ultrasonic Sensor)
    if presence_data.get('label') == 'Imposter':
        send_email_alert("Imposter Detected", f"Imposter detected via camera! Confidence: {presence_data.get('confidence')}%")
    elif isinstance(sensor_data.get('distance'), (int, float)) and sensor_data['distance'] < 50:
        send_email_alert("Imposter Detected", f"Motion nearby! Object distance: {sensor_data['distance']} cm")

def sensor_loop():
    global latest_sensor_data, latest_crop_status, latest_presence_status
    while True:
        readings = sensors.get_all_readings()
        
        # Predict Water Quality
        water_quality = water_ml.predict_water_quality(readings['tds'], readings['water_temperature'])
        readings['water_quality'] = water_quality
        
        latest_sensor_data = readings
        
        # Check alerts
        check_alerts(latest_sensor_data, latest_crop_status, latest_presence_status)
        
        time.sleep(2)

def crop_inference_loop():
    global latest_crop_status, latest_presence_status
    cam = get_camera()
    while True:
        if cam is not None and cam.isOpened():
            success, frame = cam.read()
            if success:
                # Run crop prediction
                result = crop_cnn.predict_crop_disease(frame)
                if 'confidence' in result:
                    result['confidence'] = round(result['confidence'] * 100, 1)
                latest_crop_status = result
                
                # Run presence prediction
                presence_result = presence_classifier.predict_presence(frame)
                if 'confidence' in presence_result:
                    presence_result['confidence'] = round(presence_result['confidence'] * 100, 1)
                latest_presence_status = presence_result
        else:
             # Mock result if no camera
             import random
             latest_crop_status = {
                 "label": random.choice(["Healthy", "Healthy", "Diseased", "No Plant"]), 
                 "confidence": random.randint(80, 99)
             }
             latest_presence_status = {
                 "label": random.choice(["Crop", "Human", "Imposter"]),
                 "confidence": random.randint(80, 99)
             }
        time.sleep(3)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors')
def api_sensors():
    return jsonify(latest_sensor_data)

@app.route('/api/crop_status')
def api_crop_status():
    return jsonify(latest_crop_status)

@app.route('/api/presence_status')
def api_presence_status():
    return jsonify(latest_presence_status)

@app.route('/api/recommendations')
def api_recommendations():
    # Retrieve current sensor values or set defaults
    n = latest_sensor_data.get('nitrogen', 50)
    p = latest_sensor_data.get('phosphorus', 50)
    k = latest_sensor_data.get('potassium', 50)
    ph = latest_sensor_data.get('ph', 7.0)
    soil_m = latest_sensor_data.get('soil_moisture', 50)
    
    # 1. Crop Recommendation (Mock Logic based on NPK and pH)
    crop_rec = "Wheat"
    if n > 100 and p > 60:
        crop_rec = "Rice"
    elif ph < 6.5:
        crop_rec = "Potato"
    elif k > 70:
        crop_rec = "Tomato"
        
    # 2. Fertilizer Recommendation
    fert_rec = "N/A"
    if n < 40 and p < 40:
        fert_rec = "NPK 19:19:19"
    elif n < 40:
        fert_rec = "Urea"
    elif k < 40:
        fert_rec = "Potash MOP"
    elif p < 40:
        fert_rec = "DAP"
        
    # 3. Water Supply Recommendation
    water_req = "Normal (2-3 L/day)"
    if soil_m < 30:
        water_req = "High (5+ L/day)"
    elif soil_m > 70:
        water_req = "Low (0.5 L/day)"
        
    # 4. Current Market Price (Mocked based on Crop)
    market_price = "₹2,500/Q (Wheat)"
    if crop_rec == "Rice":
        market_price = "₹3,200/Q (Rice)"
    elif crop_rec == "Potato":
        market_price = "₹1,500/Q (Potato)"
    elif crop_rec == "Tomato":
        market_price = "₹4,000/Q (Tomato)"
        
    return jsonify({
        "crop_recommendation": crop_rec,
        "fertilizer_recommendation": fert_rec,
        "water_supply": water_req,
        "market_price": market_price,
        "crop_health": latest_crop_status.get('label', 'Waiting...'),
        "presence_status": latest_presence_status.get('label', 'Waiting...'),
        "intruder_alert": latest_presence_status.get('label') == 'Imposter' or (latest_sensor_data.get('distance', 100) < 50)
    })

if __name__ == '__main__':
    # Start background threads for sensors and ML
    t1 = threading.Thread(target=sensor_loop)
    t1.daemon = True
    t1.start()
    
    t2 = threading.Thread(target=crop_inference_loop)
    t2.daemon = True
    t2.start()
    
    print("Starting Flask Server...")
    print("Access the dashboard at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 
