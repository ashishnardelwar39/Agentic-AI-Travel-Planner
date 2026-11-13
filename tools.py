"""LangChain tool wrappers around ``services.py``.

The docstring of each tool is what the LLM reads to decide when to call it,
so they are written for the model, not just for humans.
"""

import json
from typing import List, Optional

from langchain_core.tools import tool

from . import services


def _to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def flight_search_tool(source: str, destination: str) -> str:
    """Search flights between two Indian cities (e.g. source='Delhi', destination='Goa').

    Returns direct flights with the cheapest and the fastest option. If there is no
    direct flight, returns one-stop connecting options. Call it twice for a round
    trip: once for the outbound route and once with source/destination swapped for
    the return. Prices are in INR per person.
    """
    return _to_json(services.search_flights(source, destination))


@tool
def hotel_recommendation_tool(
    city: str,
    min_stars: int = 0,
    max_price_per_night: Optional[float] = None,
    required_amenities: Optional[List[str]] = None,
    sort_by: str = "value",
) -> str:
    """Recommend hotels in a city.

    Filters: min_stars (0-5), max_price_per_night in INR, required_amenities such as
    ['wifi', 'pool', 'gym', 'spa', 'breakfast', 'parking']. sort_by is 'value'
    (best stars-per-rupee), 'stars' (highest rated) or 'price' (cheapest).
    Returns up to 3 hotels.
    """
    return _to_json(services.recommend_hotels(
        city, min_stars, max_price_per_night, required_amenities, sort_by))


@tool
def places_discovery_tool(
    city: str,
    place_types: Optional[List[str]] = None,
    min_rating: float = 0.0,
) -> str:
    """Find attractions in a city, best rated first.

    place_types can include park, lake, temple, museum, fort, beach, market,
    monument (the result lists the types available in that city). min_rating is 0-5.
    Returns up to 10 places.
    """
    return _to_json(services.discover_places(city, place_types, min_rating))


@tool
def weather_lookup_tool(city: str, start_date: str, num_days: int = 3) -> str:
    """Get the daily weather for a city from the free Open-Meteo API.

    start_date must be YYYY-MM-DD. Returns max/min temperature and a summary per
    day. If the dates are beyond the 15-day forecast window, typical weather from
    the same dates in a previous year is returned and labelled as such.
    """
    return _to_json(services.get_weather(city, start_date, num_days))


@tool
def budget_estimation_tool(
    flight_cost_per_person: float,
    hotel_price_per_night: float,
    nights: int,
    travelers: int = 1,
    daily_expense_per_person: float = 1500,
) -> str:
    """Estimate the total trip cost in INR.

    flight_cost_per_person must include ALL legs (outbound + return) for one person.
    Adds hotel (one room per two travellers) and per-day food & local travel.
    nights is days minus one.
    """
    return _to_json(services.estimate_budget(
        flight_cost_per_person, hotel_price_per_night, nights, travelers,
        daily_expense_per_person))


ALL_TOOLS = [
    flight_search_tool,
    hotel_recommendation_tool,
    places_discovery_tool,
    weather_lookup_tool,
    budget_estimation_tool,
]
