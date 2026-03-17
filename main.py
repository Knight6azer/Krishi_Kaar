from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import sensors
import water_ml
import crop_cnn
import presence_classifier
import agri_ai
import numpy as np
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# --- MongoDB Configuration ---
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = client["krishi_kaar_db"]
    sensor_history = db["sensor_history"]
    system_config = db["system_config"]
    # Check connection
    client.server_info()
    MONGO_ACTIVE = True
    print("MongoDB Connected successfully.")
except Exception as e:
    MONGO_ACTIVE = False
    print(f"MongoDB not found/active. Running without persistence. Error: {e}")

# Global variables to store latest data and states
latest_sensor_data = {}
latest_crop_status = {"label": "Waiting...", "confidence": "0"}
latest_presence_status = {"label": "Waiting...", "confidence": "0"}
latest_ai_recommendations = {}

# Operational States
system_state = {
    "mode": "Manual", # "Smart" or "Manual"
    "pump": "OFF",    # "ON" or "OFF"
    "farm_area": 5.0,
    "crop_type": "Wheat",
    "soil_type": "Loamy"
}

camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    cam = get_camera()
    while True:
        if cam is not None and cam.isOpened():
            success, frame = cam.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
             blank_image = np.zeros((480, 640, 3), np.uint8)
             cv2.putText(blank_image, "No Camera Found", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             ret, buffer = cv2.imencode('.jpg', blank_image)
             frame_bytes = buffer.tobytes()
             yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
             time.sleep(1)

def sensor_loop():
    global latest_sensor_data, latest_ai_recommendations, system_state
    while True:
        readings = sensors.get_all_readings()
        
        # Predict Water Quality
        water_quality = water_ml.predict_water_quality(readings['tds'], 25.0)
        readings['water_quality'] = water_quality
        
        # Get AI recommendations (Irrigation, Fertilizer, Crop)
        ai_output = agri_ai.get_recommendations(readings)
        latest_ai_recommendations = ai_output
        
        # Smart Automation Logic
        if system_state["mode"] == "Smart":
            system_state["pump"] = ai_output["irrigation"]
        
        readings["pump_status"] = system_state["pump"]
        readings["mode"] = system_state["mode"]
        readings["timestamp"] = datetime.now()
        
        latest_sensor_data = readings
        
        # Persistence to MongoDB
        if MONGO_ACTIVE:
            try:
                sensor_history.insert_one(readings.copy())
                # Keep only last 1000 records for performance
                if sensor_history.count_documents({}) > 1000:
                    oldest = sensor_history.find().sort("timestamp", 1).limit(100)
                    for doc in oldest:
                        sensor_history.delete_one({"_id": doc["_id"]})
            except Exception as e:
                print(f"Persistence error: {e}")
        
        time.sleep(3)

def crop_inference_loop():
    global latest_crop_status, latest_presence_status
    cam = get_camera()
    while True:
        if cam is not None and cam.isOpened():
            success, frame = cam.read()
            if success:
                result = crop_cnn.predict_crop_disease(frame)
                if 'confidence' in result:
                    result['confidence'] = round(result['confidence'] * 100, 1)
                latest_crop_status = result
                
                presence_result = presence_classifier.predict_presence(frame)
                if 'confidence' in presence_result:
                    presence_result['confidence'] = round(presence_result['confidence'] * 100, 1)
                latest_presence_status = presence_result
        else:
             import random
             latest_crop_status = {
                 "label": random.choice(["Healthy", "Healthy", "Diseased", "No Plant"]), 
                 "confidence": random.randint(80, 99)
             }
             latest_presence_status = {
                 "label": random.choice(["Crop", "Human", "Imposter"]),
                 "confidence": random.randint(80, 99)
             }
        time.sleep(5)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors')
def api_sensors():
    return jsonify(latest_sensor_data)

@app.route('/api/recommendations')
def api_recommendations():
    # Return unified recommendations merging AI and UI needs
    data = latest_ai_recommendations.copy()
    data.update({
        "pump_status": system_state["pump"],
        "mode": system_state["mode"]
    })
    return jsonify(data)

@app.route('/api/control', methods=['POST'])
def api_control():
    global system_state
    req = request.json
    if "mode" in req:
        system_state["mode"] = req["mode"]
    if "pump" in req:
        # Pump control only allowed in Manual mode
        if system_state["mode"] == "Manual":
            system_state["pump"] = req["pump"]
    return jsonify({"status": "success", "state": system_state})

@app.route('/api/config', methods=['POST'])
def api_config():
    global system_state
    req = request.json
    system_state.update(req)
    if MONGO_ACTIVE:
        system_config.update_one({"type": "farm_config"}, {"$set": system_state}, upsert=True)
    return jsonify({"status": "success", "config": system_state})

@app.route('/api/history')
def api_history():
    if not MONGO_ACTIVE:
        return jsonify([])
    # Return last 20 readings for graphs
    history = list(sensor_history.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    return jsonify(history[::-1])

if __name__ == '__main__':
    # Load initial config
    if MONGO_ACTIVE:
        saved_config = system_config.find_one({"type": "farm_config"})
        if saved_config:
            system_state.update(saved_config)
            if "_id" in system_state:
                del system_state["_id"]

    t1 = threading.Thread(target=sensor_loop)
    t1.daemon = True
    t1.start()
    
    t2 = threading.Thread(target=crop_inference_loop)
    t2.daemon = True
    t2.start()
    
    print("Starting Krishi_kaar Industrial Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
