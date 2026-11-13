# ✈️ Agentic AI Travel Planning Assistant (LangChain + Streamlit)

An AI travel agent that plans multi-day trips within India. A LangChain tool-calling
agent decides which tools to call, reasons over the results and produces a day-wise
itinerary with flight, hotel, weather, budget and a justification of its choices.

## Features

| Tool | Data source | What it does |
|------|-------------|--------------|
| `flight_search_tool` | `data/flights.json` | Filters by route, suggests cheapest / fastest, falls back to one-stop options |
| `hotel_recommendation_tool` | `data/hotels.json` | Filters by city, stars, price, amenities; ranks by value, stars or price |
| `places_discovery_tool` | `data/places.json` | Attractions by type and rating |
| `weather_lookup_tool` | Open-Meteo API (free, no key) | Live forecast, or typical weather from past years for far-away dates |
| `budget_estimation_tool` | computed | Flights + hotel (1 room / 2 travellers) + daily food & local travel |

Output: human-readable markdown **and** structured JSON (downloadable), plus an "Agent steps" tab showing every tool call.

Two planning modes in the UI:
* **AI agent** – LangChain agent with OpenAI or Anthropic (needs an API key).
* **Rule-based planner** – same tools, no LLM, no key. Good for demos and testing.

## Project structure

```
├── app.py                     # Streamlit UI
├── travel_agent/
│   ├── config.py              # paths, city coordinates, defaults
│   ├── data_loader.py         # JSON loading + error handling
│   ├── services.py            # pure business logic (flights, hotels, places, weather, budget)
│   ├── tools.py               # LangChain @tool wrappers
│   ├── agent.py               # prompt, model factory, agent runner
│   └── offline_planner.py     # no-LLM planner + markdown renderer
├── data/                      # flights.json, hotels.json, places.json
├── tests/test_services.py     # unit tests (no network or API key needed)
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Provide an API key in the sidebar, or set it beforehand:

```bash
export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY=...
```

Run the tests: `python -m unittest discover -s tests`

## Deploy to GitHub + Streamlit Community Cloud

1. Create a new GitHub repository and push this folder:
   ```bash
   git init && git add . && git commit -m "Agentic AI travel planner"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, click **Create app**, choose your repo, branch `main`,
   main file path `app.py`, and deploy.
3. In the app's **Settings → Secrets**, add your key so users need not type it:
   ```toml
   OPENAI_API_KEY = "sk-..."
   # or
   ANTHROPIC_API_KEY = "..."
   ```
   Never commit keys; `.env` and `secrets.toml` are git-ignored. Without a key the rule-based mode still works.

## Notes and limitations

* **Flight data is sparse** (30 flights): 45 of 56 city pairs are bookable (25 direct, rest one-stop).
  If a route is missing (e.g. Delhi → Goa), the plan is built without airfare and says so.
  The dataset has one fixed date per flight, so flights are matched by **route**, not travel date.
* **Places dataset is synthetic** – names like "Famous Fort" can have type "lake". Names are shown as-is.
* Hotels have stars but no separate rating field, so stars are treated as the rating.
* Weather: within 15 days → live forecast; otherwise the same dates from a previous year (clearly labelled).
* Currency is INR; trip length is 3–7 days; nights = days − 1.

## Ideas to extend

Add more cities/flights from a free API, SQLite storage with indexes, a map view, or multi-turn chat refinement.
