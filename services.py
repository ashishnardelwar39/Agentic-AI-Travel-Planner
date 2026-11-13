"""Pure business logic for the travel planner.

Everything here is plain Python (no LangChain), so it can be unit-tested and
reused by both the LLM agent (through ``tools.py``) and the offline planner.
All functions return JSON-serialisable dictionaries and never raise for
"normal" problems such as an unknown city - they return an ``error`` key instead.
"""

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from .config import (
    ARCHIVE_LAG_DAYS,
    CITY_COORDINATES,
    DEFAULT_DAILY_EXPENSE_PER_PERSON,
    FORECAST_WINDOW_DAYS,
    HTTP_TIMEOUT_SECONDS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    SUPPORTED_CITIES,
)
from .data_loader import (
    DataLoadError,
    load_flights,
    load_hotels,
    load_places,
    normalise_city,
)

logger = logging.getLogger(__name__)

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow",
    75: "Heavy snow", 80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}


def _unknown_city(city: str) -> Dict[str, Any]:
    return {"error": f"Unknown city '{city}'. Supported cities: {', '.join(SUPPORTED_CITIES)}."}


# --------------------------------------------------------------------------- #
# Flights
# --------------------------------------------------------------------------- #
def _flight_summary(flight: Dict[str, Any]) -> Dict[str, Any]:
    """Add duration and a readable departure clock time to a raw flight record."""
    departure = datetime.fromisoformat(flight["departure_time"])
    arrival = datetime.fromisoformat(flight["arrival_time"])
    return {
        "flight_id": flight["flight_id"],
        "airline": flight["airline"],
        "from": flight["from"],
        "to": flight["to"],
        "departs": departure.strftime("%H:%M"),
        "arrives": arrival.strftime("%H:%M"),
        "duration_minutes": int((arrival - departure).total_seconds() // 60),
        "price": flight["price"],
    }


def search_flights(source: str, destination: str) -> Dict[str, Any]:
    """Find flights for a route; suggest the cheapest and the fastest.

    The sample dataset has one fixed date per flight, so flights are matched by
    *route* and the date is ignored. If no direct flight exists, up to three
    cheapest one-stop combinations (A -> X -> B) are offered instead.
    """
    source, destination = normalise_city(source), normalise_city(destination)
    for city in (source, destination):
        if city not in CITY_COORDINATES:
            return _unknown_city(city)
    if source == destination:
        return {"error": "Source and destination must be different cities."}
    try:
        flights = load_flights()
    except DataLoadError as exc:
        return {"error": str(exc)}

    direct = [_flight_summary(f) for f in flights
              if f["from"] == source and f["to"] == destination]
    result: Dict[str, Any] = {"source": source, "destination": destination,
                              "direct_flights": direct, "connecting_options": []}
    if direct:
        result["cheapest"] = min(direct, key=lambda f: f["price"])
        result["fastest"] = min(direct, key=lambda f: f["duration_minutes"])
        return result

    legs_out = [_flight_summary(f) for f in flights if f["from"] == source]
    legs_in = [_flight_summary(f) for f in flights if f["to"] == destination]
    connections = []
    for first in legs_out:
        for second in legs_in:
            if first["to"] == second["from"]:
                connections.append({
                    "via": first["to"],
                    "legs": [first, second],
                    "price": first["price"] + second["price"],
                    "flying_minutes": first["duration_minutes"] + second["duration_minutes"],
                })
    connections.sort(key=lambda c: c["price"])
    result["connecting_options"] = connections[:3]
    if connections:
        result["cheapest"] = connections[0]
        result["fastest"] = min(connections, key=lambda c: c["flying_minutes"])
        result["note"] = "No direct flight in the dataset; showing one-stop options."
    else:
        result["note"] = "No direct or one-stop flight found for this route."
    return result


# --------------------------------------------------------------------------- #
# Hotels
# --------------------------------------------------------------------------- #
def recommend_hotels(
    city: str,
    min_stars: int = 0,
    max_price_per_night: Optional[float] = None,
    required_amenities: Optional[List[str]] = None,
    sort_by: str = "value",
    limit: int = 3,
) -> Dict[str, Any]:
    """Filter hotels in ``city`` and rank them.

    ``sort_by``: ``"value"`` (stars per rupee, default), ``"stars"`` (highest
    rated first) or ``"price"`` (cheapest first).
    """
    city = normalise_city(city)
    if city not in CITY_COORDINATES:
        return _unknown_city(city)
    try:
        hotels = load_hotels()
    except DataLoadError as exc:
        return {"error": str(exc)}

    wanted = {a.strip().lower() for a in (required_amenities or []) if a.strip()}
    matches = []
    for hotel in hotels:
        if hotel["city"] != city or hotel["stars"] < min_stars:
            continue
        if max_price_per_night is not None and hotel["price_per_night"] > max_price_per_night:
            continue
        if not wanted.issubset(set(hotel["amenities"])):
            continue
        matches.append(hotel)

    sorters = {
        "value": lambda h: (-(h["stars"] / h["price_per_night"]), -h["stars"]),
        "stars": lambda h: (-h["stars"], h["price_per_night"]),
        "price": lambda h: (h["price_per_night"], -h["stars"]),
    }
    matches.sort(key=sorters.get(sort_by, sorters["value"]))
    result = {"city": city, "total_matches": len(matches), "hotels": matches[:limit]}
    if not matches:
        result["note"] = "No hotel matched. Try lowering min_stars, raising the price cap or dropping amenities."
    return result


# --------------------------------------------------------------------------- #
# Places
# --------------------------------------------------------------------------- #
def discover_places(
    city: str,
    place_types: Optional[List[str]] = None,
    min_rating: float = 0.0,
    limit: int = 10,
) -> Dict[str, Any]:
    """Return attractions in ``city`` ordered by rating (best first)."""
    city = normalise_city(city)
    if city not in CITY_COORDINATES:
        return _unknown_city(city)
    try:
        places = load_places()
    except DataLoadError as exc:
        return {"error": str(exc)}

    types = {t.strip().lower() for t in (place_types or []) if t.strip()}
    matches = [p for p in places
               if p["city"] == city
               and p["rating"] >= min_rating
               and (not types or p["type"].lower() in types)]
    matches.sort(key=lambda p: -p["rating"])
    available_types = sorted({p["type"] for p in places if p["city"] == city})
    result = {"city": city, "available_types": available_types,
              "total_matches": len(matches), "places": matches[:limit]}
    if not matches:
        result["note"] = "No place matched the filters; try other types or a lower rating."
    return result


# --------------------------------------------------------------------------- #
# Weather (Open-Meteo, no API key)
# --------------------------------------------------------------------------- #
def _fetch_daily_weather(url: str, latitude: float, longitude: float,
                         start: date, end: date, archive: bool) -> List[Dict[str, Any]]:
    """Call Open-Meteo and return one dict per day."""
    rain_field = "precipitation_sum" if archive else "precipitation_probability_max"
    params = {
        "latitude": latitude, "longitude": longitude,
        "daily": f"weather_code,temperature_2m_max,temperature_2m_min,{rain_field}",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    daily = response.json()["daily"]
    days = []
    for i, day in enumerate(daily["time"]):
        entry = {
            "date": day,
            "summary": WEATHER_CODES.get(daily["weather_code"][i], "Unknown"),
            "max_temp_c": daily["temperature_2m_max"][i],
            "min_temp_c": daily["temperature_2m_min"][i],
        }
        if archive:
            entry["rain_mm"] = daily[rain_field][i]
        else:
            entry["rain_chance_percent"] = daily[rain_field][i]
        days.append(entry)
    return days


def get_weather(city: str, start_date: str, num_days: int = 3) -> Dict[str, Any]:
    """Daily weather for ``num_days`` starting at ``start_date`` (YYYY-MM-DD).

    * Dates within ~15 days: live Open-Meteo forecast.
    * Dates further ahead (or in the past): the same calendar days from an
      earlier year (Open-Meteo archive), labelled as a typical-weather estimate.
    Network failures return an ``error`` so the caller can continue without weather.
    """
    city = normalise_city(city)
    if city not in CITY_COORDINATES:
        return _unknown_city(city)
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        return {"error": f"Invalid date '{start_date}'. Use YYYY-MM-DD."}
    num_days = max(1, min(int(num_days), 16))
    end = start + timedelta(days=num_days - 1)
    latitude, longitude = CITY_COORDINATES[city]
    today = date.today()

    try:
        if today <= start and end <= today + timedelta(days=FORECAST_WINDOW_DAYS):
            days = _fetch_daily_weather(OPEN_METEO_FORECAST_URL, latitude, longitude,
                                        start, end, archive=False)
            return {"city": city, "source": "Open-Meteo live forecast", "days": days}

        years_back = 1
        while end - timedelta(days=365 * years_back) >= today - timedelta(days=ARCHIVE_LAG_DAYS):
            years_back += 1
        shift = timedelta(days=365 * years_back)
        history = _fetch_daily_weather(OPEN_METEO_ARCHIVE_URL, latitude, longitude,
                                       start - shift, end - shift, archive=True)
        days = []
        for offset, entry in enumerate(history):
            entry = dict(entry)
            entry["date"] = (start + timedelta(days=offset)).isoformat()
            entry["based_on_historical_date"] = (start - shift + timedelta(days=offset)).isoformat()
            days.append(entry)
        return {"city": city,
                "source": "Open-Meteo historical data (typical weather - travel date is outside forecast range)",
                "days": days}
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("Weather lookup failed: %s", exc)
        return {"error": f"Weather service unavailable ({exc.__class__.__name__}). Continue without weather."}


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def estimate_budget(
    flight_cost_per_person: float,
    hotel_price_per_night: float,
    nights: int,
    travelers: int = 1,
    daily_expense_per_person: float = DEFAULT_DAILY_EXPENSE_PER_PERSON,
    trip_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Sum flights + hotel + local expenses (INR).

    ``flight_cost_per_person`` should already include every leg (outbound + return).
    One room is assumed per two travellers. Local expenses (food and transport)
    are charged for each trip day (default: ``nights + 1``).
    """
    if min(flight_cost_per_person, hotel_price_per_night, daily_expense_per_person) < 0 or nights < 0:
        return {"error": "Costs and nights must not be negative."}
    travelers = max(1, int(travelers))
    days = trip_days if trip_days else nights + 1
    rooms = math.ceil(travelers / 2)
    flights_total = flight_cost_per_person * travelers
    hotel_total = hotel_price_per_night * nights * rooms
    local_total = daily_expense_per_person * days * travelers
    return {
        "flights": round(flights_total),
        "hotel": round(hotel_total),
        "food_and_local_travel": round(local_total),
        "total": round(flights_total + hotel_total + local_total),
        "assumptions": f"{travelers} traveller(s), {rooms} room(s), {nights} night(s), "
                       f"{days} day(s) at Rs {daily_expense_per_person:g}/person/day for food & local travel",
        "currency": "INR",
    }
