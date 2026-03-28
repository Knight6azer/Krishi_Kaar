"""
Krishi_Kaar — Main Application Server (Production-Grade)

Enterprise Flask application with:
- Authentication (Flask-Login + password hashing)
- MongoDB primary / SQLite fallback / in-memory last resort
- Real-time sensor data via background threads
- AI-powered crop, fertilizer, and irrigation recommendations
- Computer vision (crop disease + presence detection)
- Video streaming from camera
- Multilingual i18n API
- RESTful control endpoints
"""
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import threading
import time
import json
import sqlite3
import numpy as np
from datetime import datetime
import os

import sensors
import water_ml
import vision_models
import agri_ai
import translations
from config import Config

# ============================================================
# Flask App Setup
# ============================================================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# ============================================================
# Database Layer — MongoDB → SQLite → In-Memory (cascading fallback)
# ============================================================
MONGO_ACTIVE = False
SQLITE_ACTIVE = False
users_coll = None
sensor_history = None
system_config_coll = None

# In-memory fallbacks
mock_users = []
mock_history = []

# --- Try MongoDB ---
try:
    from pymongo import MongoClient
    client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=Config.MONGO_TIMEOUT_MS)
    db = client[Config.MONGO_DB_NAME]
    users_coll = db["users"]
    sensor_history = db["sensor_history"]
    system_config_coll = db["system_config"]
    client.server_info()
    MONGO_ACTIVE = True
    print("[DB] MongoDB connected successfully.")
except Exception as e:
    print(f"[DB] MongoDB unavailable: {e}")

# --- SQLite Fallback ---
if not MONGO_ACTIVE:
    try:
        os.makedirs(os.path.dirname(Config.SQLITE_PATH), exist_ok=True)
        _sqlite_conn = sqlite3.connect(Config.SQLITE_PATH, check_same_thread=False)
        _sqlite_lock = threading.Lock()
        _sqlite_conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT DEFAULT 'Farmer',
            experience TEXT DEFAULT 'Beginner'
        )''')
        _sqlite_conn.execute('''CREATE TABLE IF NOT EXISTS sensor_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            data TEXT
        )''')
        _sqlite_conn.execute('''CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT
        )''')
        _sqlite_conn.commit()
        SQLITE_ACTIVE = True
        print(f"[DB] SQLite fallback active at {Config.SQLITE_PATH}")
    except Exception as e:
        print(f"[DB] SQLite also failed: {e}. Using in-memory storage.")


def db_type():
    if MONGO_ACTIVE: return "mongodb"
    if SQLITE_ACTIVE: return "sqlite"
    return "memory"


# ============================================================
# Authentication
# ============================================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth_page'

@login_manager.unauthorized_handler
def unauthorized():
    """Return JSON 401 for API routes, redirect for page routes."""
    if request.path.startswith('/api/'):
        return jsonify({"error": "Authentication required"}), 401
    return redirect(url_for('auth_page'))


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data.get('_id') or user_data.get('id'))
        self.email = user_data.get('email', '')
        self.name = user_data.get('name', 'Farmer')
        self.experience = user_data.get('experience', 'Beginner')


@login_manager.user_loader
def load_user(user_id):
    try:
        if MONGO_ACTIVE:
            from bson.objectid import ObjectId
            user_data = users_coll.find_one({"_id": ObjectId(user_id)})
            return User(user_data) if user_data else None
        elif SQLITE_ACTIVE:
            with _sqlite_lock:
                cur = _sqlite_conn.execute("SELECT id, email, password, name, experience FROM users WHERE id=?", (int(user_id),))
                row = cur.fetchone()
            if row:
                return User({"id": row[0], "email": row[1], "name": row[3], "experience": row[4]})
        else:
            user_data = next((u for u in mock_users if str(u.get('_id') or u.get('id')) == user_id), None)
            return User(user_data) if user_data else None
    except Exception:
        pass
    return None


# ============================================================
# System State
# ============================================================
system_state = {
    "mode": "Manual",   # Manual / Rule / Smart
    "pump": "OFF",
    "farm_area": 5.0,
    "crop_type": "Wheat",
    "soil_type": "Loamy"
}

# Global data stores (updated by background threads)
latest_sensor_data = {"soil_moisture": 0, "air_temperature": 0, "humidity": 0, "source": "initializing"}
latest_ai_recommendations = {
    "top_crops": ["Initializing..."],
    "top_crops_detailed": [{"name": "Initializing...", "confidence": 0}],
    "crop": "Initializing...",
    "fertilizer": "Initializing...",
    "irrigation": "OFF",
    "irrigation_code": 0,
    "health_score": 50
}
latest_crop_status = {"label": "Initializing...", "confidence": 0}
latest_presence_status = {"label": "Initializing...", "confidence": 0}

# Weather cache
_weather_cache = {"data": None, "expires": 0}


# ============================================================
# Camera
# ============================================================
camera = None
_camera_lock = threading.Lock()

def get_camera():
    global camera
    with _camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
        return camera


def generate_frames():
    cam = get_camera()
    while True:
        try:
            if cam is not None and cam.isOpened():
                success, frame = cam.read()
                if success:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    continue
        except Exception:
            pass
        # Fallback: blank frame
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank, "No Camera", (220, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(blank, "Using Simulation", (190, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 1)
        ret, buffer = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1)


# ============================================================
# Background Threads
# ============================================================
def sensor_loop():
    """Continuous sensor reading + AI inference + persistence."""
    global latest_sensor_data, latest_ai_recommendations, system_state
    
    while True:
        try:
            readings = sensors.get_all_readings()
            
            # Water quality prediction
            readings['water_quality'] = water_ml.predict_water_quality(readings['tds'], readings['air_temperature'])
            
            # AI recommendations
            ai_output = agri_ai.get_recommendations(readings)
            latest_ai_recommendations = ai_output
            
            # Automation logic
            if system_state["mode"] == "Smart":
                system_state["pump"] = ai_output["irrigation"]
            elif system_state["mode"] == "Rule":
                moisture = readings.get('soil_moisture', 50)
                if moisture < 30:
                    system_state["pump"] = "ON"
                elif moisture > 60:
                    system_state["pump"] = "OFF"
            
            # Enrich readings
            readings["pump_status"] = system_state["pump"]
            readings["mode"] = system_state["mode"]
            readings["health_score"] = ai_output.get("health_score", 50)
            readings["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            latest_sensor_data = readings
            
            # Persistence
            _persist_reading(readings)
            
        except Exception as e:
            print(f"[SENSOR_LOOP] Error: {e}")
        
        time.sleep(Config.SENSOR_POLL_INTERVAL)


def _persist_reading(readings):
    """Save sensor reading to available database."""
    try:
        if MONGO_ACTIVE:
            sensor_history.insert_one(readings.copy())
            count = sensor_history.count_documents({})
            if count > Config.MAX_MONGO_HISTORY:
                oldest = sensor_history.find().sort("timestamp", 1).limit(count - Config.MAX_MONGO_HISTORY)
                for doc in oldest:
                    sensor_history.delete_one({"_id": doc["_id"]})
        elif SQLITE_ACTIVE:
            data_json = json.dumps({k: v for k, v in readings.items() if k != 'timestamp'})
            with _sqlite_lock:
                _sqlite_conn.execute(
                    "INSERT INTO sensor_history (timestamp, data) VALUES (?, ?)",
                    (readings.get('timestamp', ''), data_json)
                )
                _sqlite_conn.execute(
                    f"DELETE FROM sensor_history WHERE id NOT IN (SELECT id FROM sensor_history ORDER BY id DESC LIMIT {Config.MAX_SQLITE_HISTORY})"
                )
                _sqlite_conn.commit()
        else:
            mock_history.append(readings.copy())
            if len(mock_history) > Config.MAX_MEMORY_HISTORY:
                mock_history.pop(0)
    except Exception as e:
        print(f"[DB] Persistence error: {e}")


def crop_inference_loop():
    """Background vision inference on camera frames."""
    global latest_crop_status, latest_presence_status
    
    while True:
        try:
            cam = get_camera()
            if cam is not None and cam.isOpened():
                success, frame = cam.read()
                if success:
                    result = vision_models.predict_crop_disease(frame)
                    if 'confidence' in result:
                        result['confidence'] = round(result['confidence'] * 100, 1)
                    latest_crop_status = result
                    
                    presence = vision_models.predict_presence(frame)
                    if 'confidence' in presence:
                        presence['confidence'] = round(presence['confidence'] * 100, 1)
                    latest_presence_status = presence
            else:
                # Demo mode — cycle through realistic values
                import random
                latest_crop_status = {
                    "label": random.choice(["Healthy", "Healthy", "Healthy", "Diseased", "No Plant"]),
                    "confidence": round(random.uniform(75, 99), 1),
                    "demo": True
                }
                latest_presence_status = {
                    "label": random.choice(["Crop", "Crop", "Crop", "Human"]),
                    "confidence": round(random.uniform(80, 99), 1),
                    "demo": True
                }
        except Exception as e:
            print(f"[VISION] Error: {e}")
        
        time.sleep(Config.VISION_POLL_INTERVAL)


# ============================================================
# Weather
# ============================================================
def get_weather():
    """Get weather data (cached). Uses OWM API if key provided, otherwise realistic stub."""
    global _weather_cache
    now = time.time()
    
    if _weather_cache["data"] and now < _weather_cache["expires"]:
        return _weather_cache["data"]
    
    weather = None
    
    if Config.WEATHER_API_KEY:
        try:
            import requests
            url = f"https://api.openweathermap.org/data/2.5/weather?q={Config.WEATHER_CITY}&appid={Config.WEATHER_API_KEY}&units=metric"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                d = resp.json()
                weather = {
                    "temp": round(d["main"]["temp"], 1),
                    "condition": d["weather"][0]["description"].title(),
                    "humidity": d["main"]["humidity"],
                    "wind": round(d.get("wind", {}).get("speed", 0), 1),
                    "city": Config.WEATHER_CITY,
                    "source": "openweathermap"
                }
        except Exception as e:
            print(f"[WEATHER] API error: {e}")
    
    if weather is None:
        # Realistic stub based on readings
        temp = latest_sensor_data.get("air_temperature", 0)
        hum = latest_sensor_data.get("humidity", 0)
        # Fallback defaults for initial state before sensors populate
        if temp == 0: temp = 28.0
        if hum == 0: hum = 60.0
        if hum > 80:
            condition = "Overcast"
        elif hum > 65:
            condition = "Partly Cloudy"
        else:
            condition = "Clear Sky"
        weather = {
            "temp": round(temp, 1),
            "condition": condition,
            "humidity": round(hum, 1),
            "wind": round(12 + (temp - 25) * 0.5, 1),
            "city": Config.WEATHER_CITY,
            "source": "simulated"
        }
    
    _weather_cache = {"data": weather, "expires": now + Config.WEATHER_CACHE_SEC}
    return weather


# ============================================================
# Routes — Authentication
# ============================================================
@app.route('/auth')
def auth_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('auth.html')


@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    
    if not email or not password:
        return redirect(url_for('auth_page', error="Please fill all fields"))
    
    user_data = None
    if MONGO_ACTIVE:
        user_data = users_coll.find_one({"email": email})
    elif SQLITE_ACTIVE:
        with _sqlite_lock:
            cur = _sqlite_conn.execute("SELECT id, email, password, name, experience FROM users WHERE email=?", (email,))
            row = cur.fetchone()
        if row:
            user_data = {"id": row[0], "email": row[1], "password": row[2], "name": row[3], "experience": row[4]}
    else:
        user_data = next((u for u in mock_users if u['email'] == email), None)
    
    if user_data and check_password_hash(user_data['password'], password):
        login_user(User(user_data))
        return redirect(url_for('index'))
    
    return redirect(url_for('auth_page', error="Invalid email or password"))


@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    name = request.form.get('name', 'Farmer').strip()
    experience = request.form.get('experience', 'Beginner')
    
    if not email or not password:
        return redirect(url_for('auth_page', error="Please fill all fields"))
    
    if len(password) < 4:
        return redirect(url_for('auth_page', error="Password must be at least 4 characters"))
    
    # Check existence
    existing = None
    if MONGO_ACTIVE:
        existing = users_coll.find_one({"email": email})
    elif SQLITE_ACTIVE:
        with _sqlite_lock:
            cur = _sqlite_conn.execute("SELECT id FROM users WHERE email=?", (email,))
            existing = cur.fetchone()
    else:
        existing = next((u for u in mock_users if u['email'] == email), None)
    
    if existing:
        return redirect(url_for('auth_page', error="Email already exists"))
    
    hashed_pw = generate_password_hash(password)
    
    if MONGO_ACTIVE:
        users_coll.insert_one({
            "email": email, "password": hashed_pw,
            "name": name, "experience": experience
        })
    elif SQLITE_ACTIVE:
        with _sqlite_lock:
            _sqlite_conn.execute(
                "INSERT INTO users (email, password, name, experience) VALUES (?, ?, ?, ?)",
                (email, hashed_pw, name, experience)
            )
            _sqlite_conn.commit()
    else:
        mock_users.append({
            "_id": str(datetime.now().timestamp()),
            "email": email, "password": hashed_pw,
            "name": name, "experience": experience
        })
    
    return redirect(url_for('auth_page', success="Account created! Please login."))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth_page'))


# ============================================================
# Routes — Pages
# ============================================================
@app.route('/')
@login_required
def index():
    return render_template('dashboard.html', user=current_user)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# API Endpoints
# ============================================================
@app.route('/api/sensors')
def api_sensors():
    data = latest_sensor_data.copy()
    data['db_type'] = db_type()
    data['db_connected'] = MONGO_ACTIVE or SQLITE_ACTIVE
    return jsonify(data)


@app.route('/api/recommendations')
@login_required
def api_recommendations():
    data = latest_ai_recommendations.copy()
    data.update({
        "pump_status": system_state["pump"],
        "mode": system_state["mode"],
        "user_exp": current_user.experience,
        "db_type": db_type(),
        "db_connected": MONGO_ACTIVE or SQLITE_ACTIVE
    })
    return jsonify(data)


@app.route('/api/vision')
def api_vision():
    return jsonify({
        "crop_status": latest_crop_status,
        "presence_status": latest_presence_status
    })


@app.route('/api/weather')
def api_weather():
    return jsonify(get_weather())


@app.route('/api/translations/<lang>')
def get_translations(lang):
    return jsonify(translations.translations.get(lang, translations.translations['en']))


@app.route('/api/control', methods=['POST'])
@login_required
def api_control():
    global system_state
    req = request.json or {}
    
    if "mode" in req and req["mode"] in ("Manual", "Rule", "Smart"):
        system_state["mode"] = req["mode"]
    
    if "pump" in req and req["pump"] in ("ON", "OFF"):
        if system_state["mode"] == "Manual":
            system_state["pump"] = req["pump"]
        else:
            return jsonify({"status": "error", "message": "Pump control only available in Manual mode"}), 400
    
    return jsonify({"status": "success", "state": system_state})


@app.route('/api/config', methods=['POST'])
@login_required
def api_config():
    global system_state
    req = request.json or {}
    
    allowed_keys = {"farm_area", "crop_type", "soil_type"}
    updates = {k: v for k, v in req.items() if k in allowed_keys}
    system_state.update(updates)
    
    if MONGO_ACTIVE:
        try:
            system_config_coll.update_one({"type": "farm_config"}, {"$set": system_state}, upsert=True)
        except Exception:
            pass
    elif SQLITE_ACTIVE:
        try:
            with _sqlite_lock:
                _sqlite_conn.execute(
                    "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                    ("farm_config", json.dumps(system_state))
                )
                _sqlite_conn.commit()
        except Exception:
            pass
    
    return jsonify({"status": "success", "config": system_state})


@app.route('/api/history')
def api_history():
    try:
        if MONGO_ACTIVE:
            history = list(sensor_history.find({}, {"_id": 0}).sort("timestamp", -1).limit(30))
            return jsonify(history[::-1])
        elif SQLITE_ACTIVE:
            with _sqlite_lock:
                cur = _sqlite_conn.execute(
                    "SELECT timestamp, data FROM sensor_history ORDER BY id DESC LIMIT 30"
                )
                rows = cur.fetchall()
            result = []
            for ts, data_json in reversed(rows):
                entry = json.loads(data_json)
                entry['timestamp'] = ts
                result.append(entry)
            return jsonify(result)
        else:
            return jsonify(mock_history[-30:])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/system_status')
def api_system_status():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "db": db_type(),
        "sensor_source": latest_sensor_data.get("source", "unknown"),
        "vision_demo": latest_crop_status.get("demo", False),
        "uptime": "active"
    })


# ============================================================
# Startup
# ============================================================
def _load_saved_config():
    """Load farm configuration from database on startup."""
    global system_state
    try:
        if MONGO_ACTIVE:
            saved = system_config_coll.find_one({"type": "farm_config"})
            if saved:
                for k in ("mode", "pump", "farm_area", "crop_type", "soil_type"):
                    if k in saved:
                        system_state[k] = saved[k]
        elif SQLITE_ACTIVE:
            with _sqlite_lock:
                cur = _sqlite_conn.execute("SELECT value FROM system_config WHERE key='farm_config'")
                row = cur.fetchone()
            if row:
                saved = json.loads(row[0])
                for k in ("mode", "pump", "farm_area", "crop_type", "soil_type"):
                    if k in saved:
                        system_state[k] = saved[k]
    except Exception as e:
        print(f"[STARTUP] Config load error: {e}")


if __name__ == '__main__':
    _load_saved_config()
    
    # Start background threads
    t1 = threading.Thread(target=sensor_loop, name="SensorThread", daemon=True)
    t1.start()
    
    t2 = threading.Thread(target=crop_inference_loop, name="VisionThread", daemon=True)
    t2.start()
    
    print(f"[KRISHI_KAAR] Server starting on {Config.HOST}:{Config.PORT}")
    print(f"[KRISHI_KAAR] Database: {db_type()}")
    print(f"[KRISHI_KAAR] Dashboard: http://localhost:{Config.PORT}/")
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, use_reloader=False)
