from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import datetime
import math
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LOAD MODEL
model = joblib.load("lightgbm_safe_speed_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

OPENWEATHER_API_KEY = "3864ebcddc22960f45e9a5abf09d137c"

# ROUTE ENDPOINT COORDS
KADUWELA = (6.9340, 79.9840)
KOLLUPITIYA = (6.9100, 79.8520)

# store trip start per vehicle
# WARNING: This is not production-ready. In multi-worker environments (Gunicorn, uWSGI),
# use Redis or similar external store for shared state across workers.
trip_start_location = {}

# HELPERS


def haversine_distance(coord1, coord2):
    
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def determine_direction(vehicle_id, lat, lon):
    if vehicle_id not in trip_start_location:
        trip_start_location[vehicle_id] = (lat, lon)

    start = trip_start_location[vehicle_id]

    if haversine_distance(start, KADUWELA) < haversine_distance(start, KOLLUPITIYA):
        return "Kaduwela_to_Kollupitiya"
    else:
        return "Kollupitiya_to_Kaduwela"


@lru_cache(maxsize=256)
def reverse_geocode_cached(lat_rounded, lon_rounded):
    """Cached reverse geocoding to reduce API calls.
    WARNING: Nominatim has strict usage policies. High-frequency calls will result in IP ban.
    Consider using a local geocoding database or map matching for production.
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat_rounded}&lon={lon_rounded}&format=json"
        headers = {"User-Agent": "SafeSpeedSystem/1.0"}
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
    except requests.exceptions.Timeout:
        logger.warning(f"Geocoding timeout for ({lat_rounded}, {lon_rounded})")
        return "Unknown"
    except requests.exceptions.RequestException as e:
        logger.error(f"Geocoding error: {e}")
        return "Unknown"
    except Exception as e:
        logger.error(f"Unexpected geocoding error: {e}")
        return "Unknown"


def reverse_geocode(lat, lon):
    """Wrapper that rounds coordinates for caching (approx 1.1km precision)."""
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    return reverse_geocode_cached(lat_rounded, lon_rounded)


@lru_cache(maxsize=128)
def get_weather_cached(lat_rounded, lon_rounded):
    """Cached weather data to reduce API calls and respect rate limits.
    Cache is based on rounded coordinates (approx 1.1km precision).
    Weather doesn't change significantly in small areas or short time periods.
    """
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
    except requests.exceptions.Timeout:
        logger.warning(
            f"Weather API timeout for ({lat_rounded}, {lon_rounded})")
        return 30.0, 75.0, 0.0  # Default fallback values
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API error: {e}")
        return 30.0, 75.0, 0.0  # Default fallback values
    except KeyError as e:
        logger.error(f"Weather API response missing expected field: {e}")
        return 30.0, 75.0, 0.0  # Default fallback values
    except Exception as e:
        logger.error(f"Unexpected weather API error: {e}")
        return 30.0, 75.0, 0.0  # Default fallback values


def get_weather(lat, lon):
    """Wrapper that rounds coordinates for caching."""
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    return get_weather_cached(lat_rounded, lon_rounded)


def map_rain_intensity(rain_mm):
    if rain_mm == 0:
        return 0
    elif rain_mm < 2:
        return 1
    else:
        return 2


def infer_road_condition(rain_intensity, humidity):
    return 1 if (rain_intensity > 0 or humidity >= 80) else 0


def safe_encode(col, value):
    enc = label_encoders[col]
    return int(enc.transform([value])[0]) if value in enc.classes_ else 0


#   API
@app.route("/predict", methods=["POST"])
def predict():
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
        lat = data["gps_latitude"]
        lon = data["gps_longitude"]
        passenger_count = data["passenger_count"]
        passenger_load_kg = data["passenger_load_kg"]

        # -------- AUTO DIRECTION --------
        direction = determine_direction(vehicle_id, lat, lon)

        # -------- LOCATION & WEATHER --------
        location_name = reverse_geocode(lat, lon)
        temp, humidity, rain_mm = get_weather(lat, lon)

        rain_intensity = map_rain_intensity(rain_mm)
        road_condition = infer_road_condition(rain_intensity, humidity)

        # -------- TIME FEATURES --------
        now = datetime.datetime.now()
        hour = now.hour
        day = now.weekday()
        month = now.month

        is_weekend = 1 if day >= 5 else 0
        is_peak = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
        season = 0 if month in [12, 1, 2] else 1

        # -------- DATAFRAME --------
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

        df["is_night"] = ((df["hour_of_day"] < 5) | (
            df["hour_of_day"] > 20)).astype(int)
        df["load_per_passenger"] = df["passenger_load_kg"] / \
            (df["passenger_count"] + 1)

        for col in ["vehicle_id", "route_id", "direction", "location_name"]:
            df[col] = df[col].apply(lambda x: safe_encode(col, x))

        speed = float(model.predict(df)[0])

        return jsonify({
            "safe_speed": round(speed, 1),

            "location_name": location_name,
        })

    except KeyError as e:
        logger.error(f"Missing field in request: {e}")
        return jsonify({"error": f"Missing or invalid field: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction"}), 500


if __name__ == "__main__":
    logger.info("Starting Flask API server...")
    app.run(host="0.0.0.0", port=5000)
