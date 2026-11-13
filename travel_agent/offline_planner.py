"""Rule-based planner that works without an LLM or API key.

It calls the same ``services`` functions as the LangChain tools, so the app is
usable (and demo-able) even when no model key is configured. The output has the
same structure the LLM agent is asked to produce.
"""

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from . import services
from .config import DEFAULT_DAILY_EXPENSE_PER_PERSON, MAX_PLACES_PER_DAY


def _leg_price(option: Dict[str, Any]) -> float:
    return option["price"]


def _describe_flight(option: Dict[str, Any]) -> str:
    if "legs" in option:  # one-stop
        legs = " -> ".join(f"{leg['airline']} {leg['from']}-{leg['to']}" for leg in option["legs"])
        return f"{legs} (via {option['via']})"
    return f"{option['airline']} {option['from']} -> {option['to']}, departs {option['departs']}"


def plan_trip(
    origin: str,
    destination: str,
    start_date: date,
    num_days: int,
    travelers: int = 1,
    budget: Optional[float] = None,
    min_hotel_stars: int = 0,
    place_types: Optional[List[str]] = None,
    required_amenities: Optional[List[str]] = None,
    daily_expense_per_person: float = DEFAULT_DAILY_EXPENSE_PER_PERSON,
) -> Dict[str, Any]:
    """Build a complete itinerary dictionary (or ``{"error": ...}``)."""
    nights = num_days - 1
    reasoning: List[str] = []

    # --- Flights -------------------------------------------------------- #
    outbound = services.search_flights(origin, destination)
    if "error" in outbound:
        return {"error": outbound["error"]}
    flight_out = outbound.get("cheapest")
    flight_back = None
    if flight_out is None:
        flight_per_person = 0
        reasoning.append(f"Flights: no direct or one-stop flight for {origin} -> {destination} exists in the "
                         "dataset, so the budget EXCLUDES airfare.")
    else:
        reasoning.append(f"Outbound: picked the cheapest of {len(outbound['direct_flights']) or len(outbound['connecting_options'])} "
                         f"option(s) for {origin} -> {destination}.")
        flight_back = services.search_flights(destination, origin).get("cheapest")
        if flight_back:
            return_price = _leg_price(flight_back)
            reasoning.append("Return: picked the cheapest available flight.")
        else:
            return_price = _leg_price(flight_out)
            reasoning.append("Return: no flight found in the dataset, so the outbound fare is used as an estimate.")
        flight_per_person = _leg_price(flight_out) + return_price

    # --- Hotel ---------------------------------------------------------- #
    travelers = max(1, travelers)
    rooms = math.ceil(travelers / 2)
    max_hotel_price = None
    if budget:
        local = daily_expense_per_person * num_days * travelers
        remaining = budget - flight_per_person * travelers - local
        if nights > 0 and remaining > 0:
            max_hotel_price = remaining / (nights * rooms)
    hotels = services.recommend_hotels(destination, min_hotel_stars, max_hotel_price,
                                       required_amenities, sort_by="value", limit=3)
    if not hotels.get("hotels") and max_hotel_price is not None:
        reasoning.append("No hotel fits the budget cap; ignoring the cap for the hotel search.")
        hotels = services.recommend_hotels(destination, min_hotel_stars, None,
                                           required_amenities, sort_by="price", limit=3)
    if not hotels.get("hotels"):
        return {"error": hotels.get("note", "No hotel found.")}
    hotel = hotels["hotels"][0]
    reasoning.append(f"Hotel: {hotel['name']} gives the best stars-per-rupee among {hotels['total_matches']} match(es).")

    # --- Weather -------------------------------------------------------- #
    weather = services.get_weather(destination, start_date.isoformat(), num_days)
    weather_days = weather.get("days", [])

    # --- Places & day-wise plan ---------------------------------------- #
    places_result = services.discover_places(destination, place_types, min_rating=0, limit=50)
    places = places_result.get("places", [])
    minimum_needed = min(num_days * 2, 10)
    if place_types and len(places) < minimum_needed:
        chosen = {p["place_id"] for p in places}
        extras = [p for p in services.discover_places(destination, None, 0, 50).get("places", [])
                  if p["place_id"] not in chosen]
        places = places + extras
        reasoning.append(f"Places: too few matches for your interests {place_types}; "
                         "added the best-rated other attractions to fill the days.")
    per_day = max(1, min(MAX_PLACES_PER_DAY, math.ceil(len(places) / num_days))) if places else 0
    itinerary = []
    for index in range(num_days):
        current = start_date + timedelta(days=index)
        todays = places[index * per_day:(index + 1) * per_day]
        if index == 0:
            theme = f"Arrive in {destination}"
        elif index == num_days - 1:
            theme = f"Final day and return to {origin}"
        else:
            theme = "Sightseeing"
        itinerary.append({
            "day": index + 1,
            "date": current.isoformat(),
            "theme": theme,
            "weather": weather_days[index] if index < len(weather_days) else None,
            "places": [{"name": p["name"], "type": p["type"], "rating": p["rating"]} for p in todays],
        })
    if places:
        reasoning.append(f"Places: top-rated attractions first ({per_day} per day).")

    # --- Budget --------------------------------------------------------- #
    cost = services.estimate_budget(flight_per_person, hotel["price_per_night"], nights,
                                    travelers, daily_expense_per_person, trip_days=num_days)
    if budget:
        cost["user_budget"] = budget
        cost["within_budget"] = cost["total"] <= budget

    return {
        "summary": {
            "origin": origin, "destination": destination,
            "start_date": start_date.isoformat(),
            "end_date": (start_date + timedelta(days=num_days - 1)).isoformat(),
            "days": num_days, "nights": nights, "travelers": travelers,
        },
        "flight_selected": {
            "outbound": flight_out, "return": flight_back,
            "outbound_text": _describe_flight(flight_out) if flight_out else "No flight available in the dataset for this route",
            "return_text": (_describe_flight(flight_back) if flight_back
                            else "Not in dataset (outbound fare assumed)" if flight_out else "-"),
        },
        "hotel_recommended": hotel,
        "alternative_hotels": hotels["hotels"][1:],
        "itinerary": itinerary,
        "weather_source": weather.get("source") or weather.get("error"),
        "budget": cost,
        "reasoning": reasoning,
    }


def render_markdown(plan: Dict[str, Any]) -> str:
    """Convert the dictionary from :func:`plan_trip` into readable markdown."""
    if "error" in plan:
        return f"**Could not plan this trip:** {plan['error']}"
    s, hotel, cost = plan["summary"], plan["hotel_recommended"], plan["budget"]
    lines = [
        f"## Your {s['days']}-Day Trip to {s['destination']} ({s['start_date']} to {s['end_date']})",
        f"{s['travelers']} traveller(s), departing from {s['origin']}.",
        "",
        "### Flight selected",
        f"- Outbound: {plan['flight_selected']['outbound_text']}"
        + (f" (Rs {plan['flight_selected']['outbound']['price']:,}/person)" if plan['flight_selected']['outbound'] else ""),
        f"- Return: {plan['flight_selected']['return_text']}",
        "",
        "### Hotel recommended",
        f"- **{hotel['name']}** - {hotel['stars']}-star, Rs {hotel['price_per_night']:,}/night",
        f"- Amenities: {', '.join(hotel['amenities'])}",
        "",
        "### Day-wise itinerary and weather",
    ]
    for day in plan["itinerary"]:
        weather = day["weather"]
        weather_text = (f"{weather['summary']}, {weather['min_temp_c']:.0f}-{weather['max_temp_c']:.0f} C"
                        if weather else "weather unavailable")
        lines.append(f"**Day {day['day']} - {day['date']}: {day['theme']}** ({weather_text})")
        if day["places"]:
            lines += [f"- {p['name']} ({p['type']}, rating {p['rating']})" for p in day["places"]]
        else:
            lines.append("- Free day / no attractions in the dataset")
        lines.append("")
    if plan.get("weather_source"):
        lines.append(f"_Weather source: {plan['weather_source']}_")
        lines.append("")
    lines += [
        "### Budget estimate (INR)",
        f"- Flights: Rs {cost['flights']:,}",
        f"- Hotel: Rs {cost['hotel']:,}",
        f"- Food & local travel: Rs {cost['food_and_local_travel']:,}",
        f"- **Total: Rs {cost['total']:,}**",
        f"- _Assumptions: {cost['assumptions']}_",
    ]
    if "within_budget" in cost:
        verdict = "within" if cost["within_budget"] else "over"
        lines.append(f"- This is {verdict} your budget of Rs {cost['user_budget']:,.0f}.")
    lines += ["", "### Why these choices?"] + [f"- {r}" for r in plan["reasoning"]]
    return "\n".join(lines)
