"""Streamlit front-end for the Agentic AI Travel Planning Assistant.

Run locally:  streamlit run app.py
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the `travel_agent` package importable no matter where Streamlit is launched from
# (fixes ModuleNotFoundError on Streamlit Cloud). Covers both layouts:
#   <repo>/travel_agent/...        and        <repo>/src/travel_agent/...
_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from travel_agent.config import (
    DEFAULT_DAILY_EXPENSE_PER_PERSON,
    DEFAULT_MODELS,
    MAX_TRIP_DAYS,
    MIN_TRIP_DAYS,
    SUPPORTED_CITIES,
)
from travel_agent.offline_planner import plan_trip, render_markdown

PLACE_TYPES = ["beach", "fort", "lake", "market", "monument", "museum", "park", "temple"]
AMENITIES = ["wifi", "pool", "gym", "spa", "breakfast", "parking"]
MODE_AGENT = "AI agent (LangChain + LLM)"
MODE_OFFLINE = "Rule-based planner (no API key)"

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")


def _secret(name: str) -> str:
    """Read a key from the environment or Streamlit secrets (empty if absent)."""
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return st.secrets.get(name, "")
    except Exception:  # no secrets file configured
        return ""


def build_request_text(form: dict) -> str:
    """Turn the form values into the natural-language request for the agent."""
    end = form["start_date"] + timedelta(days=form["days"] - 1)
    parts = [
        f"Plan a {form['days']}-day trip from {form['origin']} to {form['destination']} "
        f"from {form['start_date']} to {end} for {form['travelers']} traveller(s).",
        f"Assume Rs {form['daily_expense']} per person per day for food and local travel.",
    ]
    if form["budget"]:
        parts.append(f"My total budget is Rs {form['budget']}.")
    if form["min_stars"]:
        parts.append(f"Hotel should be at least {form['min_stars']} stars.")
    if form["amenities"]:
        parts.append(f"Hotel must have: {', '.join(form['amenities'])}.")
    if form["place_types"]:
        parts.append(f"I am interested in these kinds of places: {', '.join(form['place_types'])}.")
    return " ".join(parts)


# ------------------------------- Sidebar ---------------------------------- #
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Planning mode", [MODE_AGENT, MODE_OFFLINE])
    provider = api_key = model_name = ""
    if mode == MODE_AGENT:
        provider = st.selectbox("LLM provider", list(DEFAULT_MODELS))
        env_name = "OPENAI_API_KEY" if provider == "OpenAI" else "ANTHROPIC_API_KEY"
        api_key = st.text_input(f"{provider} API key", type="password",
                                value=_secret(env_name),
                                help=f"Or set {env_name} as an environment variable / Streamlit secret.")
        model_name = st.text_input("Model", value=DEFAULT_MODELS[provider])
    st.caption("Flights, hotels and places come from the bundled JSON datasets. "
               "Weather comes live from Open-Meteo.")

# -------------------------------- Main form ------------------------------- #
st.title("✈️ AI Travel Planning Assistant")
st.write("Tell the agent where you want to go. It searches flights, hotels, attractions and "
         "weather, then builds a day-wise itinerary with a budget.")

with st.form("trip_form"):
    col1, col2, col3 = st.columns(3)
    origin = col1.selectbox("From", SUPPORTED_CITIES, index=SUPPORTED_CITIES.index("Delhi"))
    destination = col2.selectbox("To", SUPPORTED_CITIES, index=SUPPORTED_CITIES.index("Goa"))
    start_date = col3.date_input("Start date", value=date.today() + timedelta(days=7),
                                 min_value=date.today())
    col4, col5, col6 = st.columns(3)
    days = col4.slider("Trip length (days)", MIN_TRIP_DAYS, MAX_TRIP_DAYS, 3)
    travelers = col5.number_input("Travellers", 1, 10, 1)
    budget = col6.number_input("Total budget (Rs, 0 = no limit)", 0, 1_000_000, 0, step=1000)
    col7, col8, col9 = st.columns(3)
    min_stars = col7.slider("Minimum hotel stars", 0, 5, 3)
    amenities = col8.multiselect("Hotel amenities", AMENITIES)
    place_types = col9.multiselect("Interests", PLACE_TYPES)
    daily_expense = st.number_input("Food & local travel per person per day (Rs)", 0, 20000,
                                    DEFAULT_DAILY_EXPENSE_PER_PERSON, step=100)
    submitted = st.form_submit_button("Plan my trip", type="primary")

# --------------------------------- Run ------------------------------------ #
if submitted:
    form = {"origin": origin, "destination": destination, "start_date": start_date,
            "days": days, "travelers": int(travelers), "budget": budget or None,
            "min_stars": min_stars, "amenities": amenities, "place_types": place_types,
            "daily_expense": daily_expense}
    st.session_state.pop("result", None)
    if origin == destination:
        st.error("Origin and destination must be different.")
    elif mode == MODE_OFFLINE:
        with st.spinner("Planning..."):
            plan = plan_trip(origin, destination, start_date, days, int(travelers),
                             budget or None, min_stars, place_types, amenities, daily_expense)
        st.session_state["result"] = {"text": render_markdown(plan), "json": plan, "trace": []}
    elif not api_key:
        st.error("Enter an API key in the sidebar, or switch to the rule-based planner.")
    else:
        try:
            from travel_agent.agent import (AgentConfigError, build_agent,
                                            extract_json_block, run_agent)
            with st.spinner("The agent is searching flights, hotels, places and weather..."):
                agent = build_agent(provider, api_key, model_name or None)
                text, trace = run_agent(agent, build_request_text(form))
            st.session_state["result"] = {"text": text, "json": extract_json_block(text),
                                          "trace": trace}
        except AgentConfigError as exc:
            st.error(str(exc))
        except Exception as exc:  # network / auth / model errors from the provider
            st.error(f"The agent failed: {exc}")

# ------------------------------- Output ----------------------------------- #
result = st.session_state.get("result")
if result:
    tab_plan, tab_json, tab_trace = st.tabs(["Itinerary", "JSON", "Agent steps"])
    with tab_plan:
        st.markdown(result["text"])
    with tab_json:
        if result["json"]:
            st.json(result["json"])
            st.download_button("Download JSON", json.dumps(result["json"], indent=2, ensure_ascii=False),
                               file_name="itinerary.json", mime="application/json")
        else:
            st.info("No structured JSON was produced for this run.")
    with tab_trace:
        if not result["trace"]:
            st.info("No tool-call trace (rule-based planner or no tools were called).")
        for item in result["trace"]:
            label = "Call" if item["step"] == "tool_call" else "Result"
            with st.expander(f"{label}: {item['tool']}"):
                st.code(json.dumps(item.get("input", item.get("output")), indent=2,
                                   ensure_ascii=False)
                        if item["step"] == "tool_call" else item["output"], language="json")
