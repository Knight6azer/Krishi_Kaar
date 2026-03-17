from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import threading
import time
import sensors
import water_ml
import crop_cnn
import presence_classifier
import agri_ai
import translations
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Weather Stub ---
def get_weather(city="Delhi"):
    # Real SaaS would use OWM API here
    return {
        "temp": 32,
        "condition": "Partly Cloudy",
        "humidity": 45,
        "wind": 12,
        "city": city
    }

# --- MongoDB Configuration ---
# In-memory fallbacks for resilient development
mock_users = []
mock_history = []
mock_config = {}

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = client["krishi_kaar_db"]
    users_coll = db["users"]
    sensor_history = db["sensor_history"]
    system_config = db["system_config"]
    # Check connection
    client.server_info()
    MONGO_ACTIVE = True
    print("MongoDB Connected successfully.")
except Exception as e:
    MONGO_ACTIVE = False
    users_coll = None
    sensor_history = None
    system_config = None
    print(f"MongoDB not found. Falling back to In-Memory storage. Error: {e}")

# --- Authentication Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth_page'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.name = user_data.get('name', 'Farmer')
        self.experience = user_data.get('experience', 'Beginner')

@login_manager.user_loader
def load_user(user_id):
    if MONGO_ACTIVE:
        from bson.objectid import ObjectId
        try:
            user_data = users_coll.find_one({"_id": ObjectId(user_id)})
            return User(user_data) if user_data else None
        except: return None
    else:
        # Fallback to mock session
        user_data = next((u for u in mock_users if str(u['_id']) == user_id), None)
        return User(user_data) if user_data else None

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
        
        # Get SaaS AI recommendations (Top crops, Health Score, etc)
        ai_output = agri_ai.get_recommendations(readings)
        latest_ai_recommendations = ai_output
        
        # Automation Mode Logic
        if system_state["mode"] == "Smart":
            system_state["pump"] = ai_output["irrigation"]
        elif system_state["mode"] == "Rule":
            # Simple Rule: Moisture < 30 triggers pump
            if readings.get('soil_moisture', 50) < 30:
                system_state["pump"] = "ON"
            elif readings.get('soil_moisture', 50) > 60:
                system_state["pump"] = "OFF"
        
        readings["pump_status"] = system_state["pump"]
        readings["mode"] = system_state["mode"]
        readings["health_score"] = ai_output.get("health_score", 50)
        readings["timestamp"] = datetime.now()
        
        latest_sensor_data = readings
        
        # Persistence to MongoDB
        if MONGO_ACTIVE:
            try:
                # Add user context if possible
                sensor_history.insert_one(readings.copy())
                if sensor_history.count_documents({}) > 1000:
                    oldest = sensor_history.find().sort("timestamp", 1).limit(100)
                    for doc in oldest: sensor_history.delete_one({"_id": doc["_id"]})
            except Exception as e:
                print(f"Persistence error: {e}")
        else:
            # SaaS Fallback: Keep last 20 in memory
            mock_history.append(readings.copy())
            if len(mock_history) > 20: mock_history.pop(0)
        
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

@app.route('/auth')
def auth_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    
    if MONGO_ACTIVE:
        user_data = users_coll.find_one({"email": email})
    else:
        user_data = next((u for u in mock_users if u['email'] == email), None)
    
    if user_data and check_password_hash(user_data['password'], password):
        user_obj = User(user_data)
        login_user(user_obj)
        return redirect(url_for('index'))
    
    return redirect(url_for('auth_page', error="Invalid email or password"))

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')
    password = request.form.get('password')
    name = request.form.get('name')
    experience = request.form.get('experience')
    
    # Check existence
    if MONGO_ACTIVE:
        existing = users_coll.find_one({"email": email})
    else:
        existing = next((u for u in mock_users if u['email'] == email), None)
        
    if existing:
        return redirect(url_for('auth_page', error="Email already exists"))
    
    hashed_pw = generate_password_hash(password, method='sha256')
    new_user = {
        "_id": datetime.now().timestamp(), # Mock ID for memory mode
        "email": email,
        "password": hashed_pw,
        "name": name,
        "experience": experience,
        "farms": [{"name": "Default Farm", "area": 5.0, "soil": "Loamy"}]
    }
    
    if MONGO_ACTIVE:
        # Pymongo will overwrite _id with ObjectId if we don't handle it
        del new_user["_id"]
        users_coll.insert_one(new_user)
    else:
        mock_users.append(new_user)
        
    flash("Account created! Please login.")
    return redirect(url_for('auth_page'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth_page'))

@app.route('/')
@login_required
def index():
    return render_template('dashboard.html', user=current_user)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors')
def api_sensors():
    return jsonify(latest_sensor_data)

@app.route('/api/translations/<lang>')
def get_translations(lang):
    return jsonify(translations.translations.get(lang, translations.translations['en']))

@app.route('/api/weather')
@login_required
def api_weather():
    return jsonify(get_weather("Bangalore"))

@app.route('/api/recommendations')
@login_required
def api_recommendations():
    data = latest_ai_recommendations.copy()
    data.update({
        "pump_status": system_state["pump"],
        "mode": system_state["mode"],
        "user_exp": current_user.experience,
        "db_connected": MONGO_ACTIVE
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
    if MONGO_ACTIVE:
        # Return last 20 readings for graphs
        history = list(sensor_history.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
        return jsonify(history[::-1])
    else:
        return jsonify(mock_history)

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
