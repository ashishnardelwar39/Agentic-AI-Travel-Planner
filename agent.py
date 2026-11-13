"""LangChain tool-calling agent that plans trips autonomously."""

import json
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_MODELS
from .tools import ALL_TOOLS

SYSTEM_PROMPT_TEMPLATE = """You are an expert travel planner for trips within India. Today's date is {today}.
Use ONLY your tools for flights, hotels, places, weather and cost - never invent data.

Work step by step:
1. Search the outbound flight, then the return flight (swap source and destination).
   Choose the cheapest unless the user prefers speed. If a route has no return flight in the data,
   say so and assume the outbound fare for the return.
2. Get hotel recommendations that respect the user's star, price and amenity preferences.
   Keep hotel price within the user's budget when one is given.
3. Find attractions that match the user's interests, best rated first.
4. Look up the weather for the exact travel dates. If the tool says the data is historical,
   tell the user it is typical weather and not a forecast. If the tool fails, continue without it.
5. Estimate the budget with the budget tool (nights = days - 1; flight cost must include both legs).
6. Build a day-wise itinerary with 2-3 places per day, no repeats, using only places the tool returned.

Final answer format (markdown, in this order):
## Your <N>-Day Trip to <City> (<dates>)
### Flight selected
### Hotel recommended
### Day-wise itinerary and weather   (one block per day with weather)
### Budget estimate (INR)           (flights, hotel, food & local travel, total, and whether it fits the budget)
### Why these choices?              (short justification for the flight, hotel and places)

Then append ONE fenced ```json block containing a compact machine-readable version with the keys:
summary, flight_selected, hotel_recommended, itinerary (list of days with date, weather, places), budget.
Amounts are in INR. If something is impossible (unknown city, no hotel), explain it clearly instead."""


class AgentConfigError(RuntimeError):
    """Raised when the LLM cannot be created (missing key, package, ...)."""


def build_chat_model(provider: str, api_key: str, model_name: Optional[str] = None):
    """Create the chat model for ``provider`` ('OpenAI' or 'Anthropic')."""
    if not api_key:
        raise AgentConfigError(f"No API key provided for {provider}.")
    model_name = model_name or DEFAULT_MODELS[provider]
    try:
        if provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
        if provider == "Anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, api_key=api_key, temperature=0)
    except ImportError as exc:
        raise AgentConfigError(
            f"Missing package for {provider}: {exc}. Run `pip install -r requirements.txt`.") from exc
    raise AgentConfigError(f"Unsupported provider: {provider}")


def build_agent(provider: str, api_key: str, model_name: Optional[str] = None):
    """Return a LangChain agent wired to all five travel tools."""
    from langchain.agents import create_agent

    model = build_chat_model(provider, api_key, model_name)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
    return create_agent(model=model, tools=ALL_TOOLS, system_prompt=system_prompt)


def _message_text(content: Any) -> str:
    """Flatten message content (a string, or a list of blocks for some providers)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Return the last ```json block in ``text`` as a dict, or None."""
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    for raw in reversed(blocks):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def run_agent(agent, user_request: str, recursion_limit: int = 30) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the agent; return (final answer text, trace of tool calls)."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_request}]},
                          config={"recursion_limit": recursion_limit})
    messages = result["messages"]
    trace: List[Dict[str, Any]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            trace.append({"step": "tool_call", "tool": call["name"], "input": call["args"]})
        if message.type == "tool":
            trace.append({"step": "tool_result", "tool": message.name,
                          "output": _message_text(message.content)})
    return _message_text(messages[-1].content), trace
