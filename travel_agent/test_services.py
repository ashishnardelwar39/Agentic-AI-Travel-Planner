"""Unit tests for the business logic. Run with:  python -m unittest discover -s tests"""

import os
import sys
import unittest
from datetime import date, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_agent import services  # noqa: E402
from travel_agent.offline_planner import plan_trip, render_markdown  # noqa: E402


class FlightTests(unittest.TestCase):
    def test_direct_route_returns_cheapest_and_fastest(self):
        result = services.search_flights("hyderabad", "Delhi")
        self.assertEqual(result["cheapest"]["flight_id"], "FL0001")
        self.assertEqual(result["cheapest"]["duration_minutes"], 240)

    def test_one_stop_when_no_direct(self):
        result = services.search_flights("Delhi", "Jaipur")
        self.assertEqual(result["direct_flights"], [])
        self.assertTrue(result.get("connecting_options") is not None)

    def test_unknown_city(self):
        self.assertIn("error", services.search_flights("Paris", "Delhi"))

    def test_same_city(self):
        self.assertIn("error", services.search_flights("Goa", "Goa"))


class HotelAndPlaceTests(unittest.TestCase):
    def test_hotel_filters(self):
        result = services.recommend_hotels("Goa", min_stars=4, max_price_per_night=3000,
                                           required_amenities=["pool"])
        self.assertTrue(result["hotels"])
        for hotel in result["hotels"]:
            self.assertEqual(hotel["city"], "Goa")
            self.assertGreaterEqual(hotel["stars"], 4)
            self.assertLessEqual(hotel["price_per_night"], 3000)
            self.assertIn("pool", hotel["amenities"])

    def test_hotel_no_match_has_note(self):
        result = services.recommend_hotels("Goa", max_price_per_night=100)
        self.assertEqual(result["hotels"], [])
        self.assertIn("note", result)

    def test_places_sorted_by_rating(self):
        ratings = [p["rating"] for p in services.discover_places("Kolkata")["places"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))

    def test_places_type_filter(self):
        result = services.discover_places("Delhi", place_types=["lake"])
        self.assertTrue(all(p["type"] == "lake" for p in result["places"]))


class BudgetTests(unittest.TestCase):
    def test_matches_project_document_example(self):
        # Doc example: flight 4800, hotel 3200 x 2 nights, food & travel 2500
        result = services.estimate_budget(4800, 3200, 2, travelers=1,
                                          daily_expense_per_person=2500 / 3, trip_days=3)
        self.assertEqual(result["total"], 4800 + 6400 + 2500)

    def test_rooms_scale_with_travelers(self):
        result = services.estimate_budget(1000, 2000, 2, travelers=3, daily_expense_per_person=0)
        self.assertEqual(result["hotel"], 2000 * 2 * 2)  # 2 rooms
        self.assertEqual(result["flights"], 3000)


class WeatherTests(unittest.TestCase):
    def test_bad_date(self):
        self.assertIn("error", services.get_weather("Goa", "not-a-date"))

    def test_network_failure_is_handled(self):
        import requests
        with patch.object(services.requests, "get", side_effect=requests.ConnectionError("down")):
            self.assertIn("error", services.get_weather("Goa", date.today().isoformat()))

    def test_forecast_parsing(self):
        payload = {"daily": {"time": ["2030-01-01"], "weather_code": [0],
                             "temperature_2m_max": [31.0], "temperature_2m_min": [22.0],
                             "precipitation_probability_max": [5]}}
        with patch.object(services.requests, "get") as mocked:
            mocked.return_value.json.return_value = payload
            mocked.return_value.raise_for_status.return_value = None
            result = services.get_weather("Goa", date.today().isoformat(), 1)
        self.assertEqual(result["days"][0]["summary"], "Clear sky")


class PlannerTests(unittest.TestCase):
    def test_full_plan_without_weather(self):
        with patch.object(services, "get_weather", return_value={"error": "offline"}):
            plan = plan_trip("Bangalore", "Goa", date.today() + timedelta(days=5), 3, travelers=2,
                             budget=60000, min_hotel_stars=3)
        self.assertNotIn("error", plan)
        self.assertEqual(len(plan["itinerary"]), 3)
        self.assertEqual(plan["summary"]["nights"], 2)
        text = render_markdown(plan)
        self.assertIn("Total", text)

    def test_missing_flight_route_still_plans_without_airfare(self):
        # Delhi -> Goa has no flight (direct or one-stop) in the sample dataset.
        with patch.object(services, "get_weather", return_value={"error": "offline"}):
            plan = plan_trip("Delhi", "Goa", date.today() + timedelta(days=5), 3)
        self.assertNotIn("error", plan)
        self.assertEqual(plan["budget"]["flights"], 0)
        self.assertIn("EXCLUDES airfare", " ".join(plan["reasoning"]))

    def test_round_trip_route(self):
        with patch.object(services, "get_weather", return_value={"error": "offline"}):
            plan = plan_trip("Bangalore", "Goa", date.today() + timedelta(days=5), 4)
        self.assertIsNotNone(plan["flight_selected"]["return"])

    def test_unknown_city_returns_error(self):
        plan = plan_trip("Delhi", "Atlantis", date.today(), 3)
        self.assertIn("error", plan)


if __name__ == "__main__":
    unittest.main()
