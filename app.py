"""
Smart Bus Safe Speed Prediction & Fleet Management System
Backend API Server with MongoDB Integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import joblib
import requests
import math
import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from bson import ObjectId
import json

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "bus_speed_predict_api")

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()  # Test connection
    db = mongo_client[DB_NAME]
    telemetry_collection = db["telemetry"]
    buses_collection = db["buses"]
    predictions_collection = db["predictions"]
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None

# Load ML Model
model_path = "lightgbm_safe_speed_model.pkl"
encoders_path = "label_encoders.pkl"

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Label encoders file not found: {encoders_path}")
    
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    logger.info("Model and label encoders loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    label_encoders = None

# API Keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Route Endpoints (for direction calculation)
ROUTE_ENDPOINTS = {
    "177_Kaduwela_Kollupitiya": {
        "start": (6.936372181, 79.98325019),  # Kaduwela
        "end": (6.91145983, 79.86845281)       # Kollupitiya
    }
}

# Store trip start per vehicle (in-memory, could be moved to Redis)
trip_start_location = {}

# ========================
# HELPER FUNCTIONS
# ========================

def haversine_distance(coord1, coord2):
    """Calculate distance between two GPS coordinates in km"""
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def determine_direction(vehicle_id, lat, lon, route_id="177_Kaduwela_Kollupitiya"):
    """Determine travel direction based on trip start location"""
    if vehicle_id not in trip_start_location:
        trip_start_location[vehicle_id] = (lat, lon)
    
    start = trip_start_location[vehicle_id]
    endpoints = ROUTE_ENDPOINTS.get(route_id, ROUTE_ENDPOINTS["177_Kaduwela_Kollupitiya"])
    
    if haversine_distance(start, endpoints["start"]) < haversine_distance(start, endpoints["end"]):
        return "Kaduwela_to_Kollupitiya"
    else:
        return "Kollupitiya_to_Kaduwela"


@lru_cache(maxsize=256)
def reverse_geocode_cached(lat_rounded, lon_rounded):
    """Reverse geocode GPS coordinates to location name (cached)"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat_rounded}&lon={lon_rounded}&format=json"
        headers = {"User-Agent": "SmartBusFleetSystem/1.0"}
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        addr = data.get("address", {})
        location = (
            addr.get("suburb")
            or addr.get("neighbourhood")
            or addr.get("town")
            or addr.get("city")
            or "Unknown"
        )
        return location
    except Exception as e:
        logger.warning(f"Geocoding error: {e}")
        return "Unknown"


def reverse_geocode(lat, lon):
    """Reverse geocode with rounding for cache efficiency"""
    lat_rounded = round(lat, 3)
    lon_rounded = round(lon, 3)
    return reverse_geocode_cached(lat_rounded, lon_rounded)


@lru_cache(maxsize=128)
def get_weather_cached(lat_rounded, lon_rounded):
    """Get weather data from OpenWeatherMap (cached)"""
    if not OPENWEATHER_API_KEY:
        return 30.0, 75.0, 0.0
    
    try:
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat_rounded}&lon={lon_rounded}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rain_mm = data.get("rain", {}).get("1h", 0)
        
        return temp, humidity, rain_mm
    except Exception as e:
        logger.warning(f"Weather API error: {e}")
        return 30.0, 75.0, 0.0


def get_weather(lat, lon):
    """Get weather with rounding for cache efficiency"""
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    return get_weather_cached(lat_rounded, lon_rounded)


def map_rain_intensity(rain_mm):
    """Map rain mm to intensity levels"""
    if rain_mm == 0:
        return 0
    elif rain_mm < 2:
        return 1
    else:
        return 2


def infer_road_condition(rain_intensity, humidity):
    """Infer road condition from weather data"""
    return 1 if (rain_intensity > 0 or humidity >= 80) else 0


def get_road_condition_label(condition):
    """Convert road condition code to label"""
    return "Wet" if condition == 1 else "Dry"


def safe_encode(col, value):
    """Safely encode categorical values"""
    if label_encoders is None:
        return 0
    enc = label_encoders.get(col)
    if enc is None:
        return 0
    return int(enc.transform([value])[0]) if value in enc.classes_ else 0


def json_serializer(obj):
    """Custom JSON serializer for MongoDB ObjectId and datetime"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ========================
# ESP32 API ENDPOINTS
# ========================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint for ESP32 devices
    Receives minimal telemetry, derives additional features, and returns safe speed
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ["vehicle_id", "route_id", "gps_latitude", "gps_longitude",
                          "passenger_count", "passenger_load_kg"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        vehicle_id = data["vehicle_id"]
        route_id = data["route_id"]
        lat = float(data["gps_latitude"])
        lon = float(data["gps_longitude"])
        passenger_count = int(data["passenger_count"])
        passenger_load_kg = float(data["passenger_load_kg"])
        
        # Derive direction
        direction = determine_direction(vehicle_id, lat, lon, route_id)
        
        # Get location and weather
        location_name = reverse_geocode(lat, lon)
        temp, humidity, rain_mm = get_weather(lat, lon)
        
        rain_intensity = map_rain_intensity(rain_mm)
        road_condition = infer_road_condition(rain_intensity, humidity)
        road_condition_label = get_road_condition_label(road_condition)
        
        # Time features
        now = datetime.now()
        hour = now.hour
        day = now.weekday()
        month = now.month
        
        is_weekend = 1 if day >= 5 else 0
        is_peak = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
        season = 0 if month in [12, 1, 2] else 1
        
        # Build feature DataFrame
        df = pd.DataFrame([{
            "vehicle_id": vehicle_id,
            "route_id": route_id,
            "direction": direction,
            "location_name": location_name,
            "gps_latitude": lat,
            "gps_longitude": lon,
            "passenger_count": passenger_count,
            "passenger_load_kg": passenger_load_kg,
            "road_condition": road_condition,
            "temperature_c": temp,
            "humidity_percent": humidity,
            "rain_intensity": rain_intensity,
            "hour_of_day": hour,
            "day_of_week": day,
            "is_weekend": is_weekend,
            "is_peak_hours": is_peak,
            "month": month,
            "season": season
        }])
        
        df["is_night"] = ((df["hour_of_day"] < 5) | (df["hour_of_day"] > 20)).astype(int)
        df["load_per_passenger"] = df["passenger_load_kg"] / (df["passenger_count"] + 1)
        
        # Encode categorical features
        for col in ["vehicle_id", "route_id", "direction", "location_name"]:
            df[col] = df[col].apply(lambda x: safe_encode(col, x))
        
        # Predict safe speed
        if model is not None:
            safe_speed = float(model.predict(df)[0])
        else:
            safe_speed = 40.0  # Default fallback
        
        safe_speed = round(safe_speed, 1)
        
        # Prepare response data
        response_data = {
            "safe_speed": safe_speed,
            "location_name": location_name,
            "road_condition": road_condition_label,
            "direction": direction,
            "temperature": temp,
            "humidity": humidity
        }
        
        # Store in database
        if db is not None:
            timestamp = now
            
            # Update or insert bus record
            bus_data = {
                "vehicle_id": vehicle_id,
                "route_id": route_id,
                "latitude": lat,
                "longitude": lon,
                "location_name": location_name,
                "direction": direction,
                "safe_speed": safe_speed,
                "road_condition": road_condition_label,
                "passenger_count": passenger_count,
                "passenger_load_kg": passenger_load_kg,
                "temperature": temp,
                "humidity": humidity,
                "last_update": timestamp,
                "status": "online"
            }
            
            buses_collection.update_one(
                {"vehicle_id": vehicle_id},
                {"$set": bus_data},
                upsert=True
            )
            
            # Store telemetry record
            telemetry_data = {
                **bus_data,
                "timestamp": timestamp
            }
            telemetry_collection.insert_one(telemetry_data)
            
            # Emit WebSocket event for real-time updates
            socketio.emit('bus_update', json.loads(json.dumps(bus_data, default=json_serializer)))
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/reset-trip/<vehicle_id>", methods=["POST"])
def reset_trip(vehicle_id):
    """Reset trip start location for a vehicle (new trip)"""
    if vehicle_id in trip_start_location:
        del trip_start_location[vehicle_id]
    return jsonify({"message": f"Trip reset for {vehicle_id}"})


# ========================
# FLEET MANAGEMENT API ENDPOINTS
# ========================

@app.route("/api/fleet/overview", methods=["GET"])
def fleet_overview():
    """Get fleet overview statistics"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        # Get total buses
        total_buses = buses_collection.count_documents({})
        
        # Get online buses (updated in last 30 seconds)
        online_threshold = datetime.now() - timedelta(seconds=30)
        online_buses = buses_collection.count_documents({
            "last_update": {"$gte": online_threshold}
        })
        
        # Get offline buses
        offline_buses = total_buses - online_buses
        
        # Get average speed
        avg_speed_result = list(buses_collection.aggregate([
            {"$group": {"_id": None, "avg_speed": {"$avg": "$safe_speed"}}}
        ]))
        avg_speed = round(avg_speed_result[0]["avg_speed"], 1) if avg_speed_result else 0
        
        # Get road condition summary
        wet_roads = buses_collection.count_documents({"road_condition": "Wet"})
        dry_roads = buses_collection.count_documents({"road_condition": "Dry"})
        
        # Get total passengers
        total_passengers_result = list(buses_collection.aggregate([
            {"$group": {"_id": None, "total_passengers": {"$sum": "$passenger_count"}}}
        ]))
        total_passengers = total_passengers_result[0]["total_passengers"] if total_passengers_result else 0
        
        return jsonify({
            "total_buses": total_buses,
            "online_buses": online_buses,
            "offline_buses": offline_buses,
            "average_speed": avg_speed,
            "road_conditions": {
                "wet": wet_roads,
                "dry": dry_roads
            },
            "total_passengers": total_passengers,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Fleet overview error: {e}")
        return jsonify({"error": "Failed to fetch fleet overview"}), 500


@app.route("/api/fleet/buses", methods=["GET"])
def get_all_buses():
    """Get list of all buses with current status"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        buses = list(buses_collection.find({}).sort("vehicle_id", 1))
        
        online_threshold = datetime.now() - timedelta(seconds=30)
        
        bus_list = []
        for bus in buses:
            is_online = bus.get("last_update", datetime.min) >= online_threshold
            bus_list.append({
                "id": str(bus["_id"]),
                "vehicle_id": bus.get("vehicle_id"),
                "route_id": bus.get("route_id"),
                "latitude": bus.get("latitude"),
                "longitude": bus.get("longitude"),
                "location_name": bus.get("location_name"),
                "direction": bus.get("direction"),
                "safe_speed": bus.get("safe_speed"),
                "road_condition": bus.get("road_condition"),
                "passenger_count": bus.get("passenger_count"),
                "passenger_load_kg": bus.get("passenger_load_kg"),
                "temperature": bus.get("temperature"),
                "humidity": bus.get("humidity"),
                "last_update": bus.get("last_update").isoformat() if bus.get("last_update") else None,
                "status": "online" if is_online else "offline"
            })
        
        return jsonify({"buses": bus_list, "count": len(bus_list)})
        
    except Exception as e:
        logger.error(f"Get buses error: {e}")
        return jsonify({"error": "Failed to fetch buses"}), 500


@app.route("/api/fleet/buses/<vehicle_id>", methods=["GET"])
def get_bus_details(vehicle_id):
    """Get detailed information about a specific bus"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        bus = buses_collection.find_one({"vehicle_id": vehicle_id})
        
        if not bus:
            return jsonify({"error": "Bus not found"}), 404
        
        online_threshold = datetime.now() - timedelta(seconds=30)
        is_online = bus.get("last_update", datetime.min) >= online_threshold
        
        return jsonify({
            "id": str(bus["_id"]),
            "vehicle_id": bus.get("vehicle_id"),
            "route_id": bus.get("route_id"),
            "latitude": bus.get("latitude"),
            "longitude": bus.get("longitude"),
            "location_name": bus.get("location_name"),
            "direction": bus.get("direction"),
            "safe_speed": bus.get("safe_speed"),
            "road_condition": bus.get("road_condition"),
            "passenger_count": bus.get("passenger_count"),
            "passenger_load_kg": bus.get("passenger_load_kg"),
            "temperature": bus.get("temperature"),
            "humidity": bus.get("humidity"),
            "last_update": bus.get("last_update").isoformat() if bus.get("last_update") else None,
            "status": "online" if is_online else "offline"
        })
        
    except Exception as e:
        logger.error(f"Get bus details error: {e}")
        return jsonify({"error": "Failed to fetch bus details"}), 500


@app.route("/api/fleet/buses/<vehicle_id>/history", methods=["GET"])
def get_bus_history(vehicle_id):
    """Get telemetry history for a specific bus"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        # Get query parameters
        limit = int(request.args.get("limit", 100))
        hours = int(request.args.get("hours", 24))
        
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        history = list(telemetry_collection.find({
            "vehicle_id": vehicle_id,
            "timestamp": {"$gte": time_threshold}
        }).sort("timestamp", -1).limit(limit))
        
        history_list = []
        for record in history:
            history_list.append({
                "timestamp": record.get("timestamp").isoformat() if record.get("timestamp") else None,
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "location_name": record.get("location_name"),
                "direction": record.get("direction"),
                "safe_speed": record.get("safe_speed"),
                "road_condition": record.get("road_condition"),
                "passenger_count": record.get("passenger_count"),
                "temperature": record.get("temperature"),
                "humidity": record.get("humidity")
            })
        
        return jsonify({
            "vehicle_id": vehicle_id,
            "history": history_list,
            "count": len(history_list)
        })
        
    except Exception as e:
        logger.error(f"Get bus history error: {e}")
        return jsonify({"error": "Failed to fetch bus history"}), 500


@app.route("/api/fleet/map-data", methods=["GET"])
def get_map_data():
    """Get data for map visualization (all buses with coordinates)"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        buses = list(buses_collection.find({
            "latitude": {"$exists": True},
            "longitude": {"$exists": True}
        }))
        
        online_threshold = datetime.now() - timedelta(seconds=30)
        
        map_data = []
        for bus in buses:
            is_online = bus.get("last_update", datetime.min) >= online_threshold
            map_data.append({
                "vehicle_id": bus.get("vehicle_id"),
                "latitude": bus.get("latitude"),
                "longitude": bus.get("longitude"),
                "location_name": bus.get("location_name"),
                "safe_speed": bus.get("safe_speed"),
                "road_condition": bus.get("road_condition"),
                "direction": bus.get("direction"),
                "passenger_count": bus.get("passenger_count"),
                "status": "online" if is_online else "offline"
            })
        
        return jsonify({"buses": map_data, "count": len(map_data)})
        
    except Exception as e:
        logger.error(f"Get map data error: {e}")
        return jsonify({"error": "Failed to fetch map data"}), 500


@app.route("/api/fleet/routes", methods=["GET"])
def get_routes():
    """Get all unique routes in the system"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        routes = buses_collection.distinct("route_id")
        
        route_details = []
        for route_id in routes:
            bus_count = buses_collection.count_documents({"route_id": route_id})
            route_details.append({
                "route_id": route_id,
                "bus_count": bus_count
            })
        
        return jsonify({"routes": route_details, "count": len(route_details)})
        
    except Exception as e:
        logger.error(f"Get routes error: {e}")
        return jsonify({"error": "Failed to fetch routes"}), 500


@app.route("/api/fleet/statistics", methods=["GET"])
def get_statistics():
    """Get detailed fleet statistics"""
    if db is None:
        return jsonify({"error": "Database not available"}), 503
    
    try:
        # Speed distribution
        speed_ranges = [
            {"label": "0-20 km/h", "min": 0, "max": 20},
            {"label": "20-40 km/h", "min": 20, "max": 40},
            {"label": "40-60 km/h", "min": 40, "max": 60},
            {"label": "60-80 km/h", "min": 60, "max": 80},
            {"label": "80+ km/h", "min": 80, "max": 200}
        ]
        
        speed_distribution = []
        for range_info in speed_ranges:
            count = buses_collection.count_documents({
                "safe_speed": {"$gte": range_info["min"], "$lt": range_info["max"]}
            })
            speed_distribution.append({
                "range": range_info["label"],
                "count": count
            })
        
        # Hourly telemetry count for today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        hourly_data = list(telemetry_collection.aggregate([
            {"$match": {"timestamp": {"$gte": today_start}}},
            {"$group": {
                "_id": {"$hour": "$timestamp"},
                "count": {"$sum": 1},
                "avg_speed": {"$avg": "$safe_speed"}
            }},
            {"$sort": {"_id": 1}}
        ]))
        
        return jsonify({
            "speed_distribution": speed_distribution,
            "hourly_telemetry": [{"hour": h["_id"], "count": h["count"], "avg_speed": round(h.get("avg_speed", 0), 1)} for h in hourly_data],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get statistics error: {e}")
        return jsonify({"error": "Failed to fetch statistics"}), 500


# ========================
# WEBSOCKET EVENTS
# ========================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to WebSocket")
    emit('connected', {'message': 'Connected to Smart Bus Fleet System'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from WebSocket")


@socketio.on('subscribe_updates')
def handle_subscribe():
    """Subscribe to real-time bus updates"""
    logger.info("Client subscribed to bus updates")
    emit('subscribed', {'message': 'Subscribed to real-time updates'})


# ========================
# HEALTH & STATUS ENDPOINTS
# ========================

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    db_status = "connected" if db is not None else "disconnected"
    model_status = "loaded" if model is not None else "not loaded"
    
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "model": model_status,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return jsonify({
        "name": "Smart Bus Safe Speed Prediction & Fleet Management System",
        "version": "1.0.0",
        "endpoints": {
            "ESP32": {
                "POST /predict": "Get safe speed prediction"
            },
            "Fleet Management": {
                "GET /api/fleet/overview": "Fleet overview statistics",
                "GET /api/fleet/buses": "List all buses",
                "GET /api/fleet/buses/<vehicle_id>": "Get bus details",
                "GET /api/fleet/buses/<vehicle_id>/history": "Get bus telemetry history",
                "GET /api/fleet/map-data": "Get data for map visualization",
                "GET /api/fleet/routes": "Get all routes",
                "GET /api/fleet/statistics": "Get detailed statistics"
            }
        }
    })


# ========================
# MAIN
# ========================

if __name__ == "__main__":
    logger.info("Starting Smart Bus Fleet Management System...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
