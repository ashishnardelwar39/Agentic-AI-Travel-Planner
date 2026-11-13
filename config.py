"""Central configuration: file locations, city coordinates and defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FLIGHTS_FILE = DATA_DIR / "flights.json"
HOTELS_FILE = DATA_DIR / "hotels.json"
PLACES_FILE = DATA_DIR / "places.json"

# Coordinates used for the Open-Meteo weather lookups.
CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Goa": (15.2993, 74.1240),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
}

SUPPORTED_CITIES = sorted(CITY_COORDINATES)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_WINDOW_DAYS = 15  # Open-Meteo forecasts reach ~16 days ahead
ARCHIVE_LAG_DAYS = 7  # archive data is complete only after a few days
HTTP_TIMEOUT_SECONDS = 10

DEFAULT_DAILY_EXPENSE_PER_PERSON = 1500  # food + local transport, in INR
MIN_TRIP_DAYS = 3
MAX_TRIP_DAYS = 7
MAX_PLACES_PER_DAY = 3

DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Anthropic": "claude-sonnet-5-5",
}
