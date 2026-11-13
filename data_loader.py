"""Loading of the JSON datasets (flights, hotels, places)."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from .config import FLIGHTS_FILE, HOTELS_FILE, PLACES_FILE

logger = logging.getLogger(__name__)


class DataLoadError(RuntimeError):
    """Raised when a dataset cannot be read or parsed."""


@lru_cache(maxsize=None)
def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON array from ``path`` (cached after the first call)."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Dataset is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, list):
        raise DataLoadError(f"Dataset must be a JSON list: {path}")
    logger.debug("Loaded %d records from %s", len(data), path.name)
    return data


def load_flights() -> List[Dict[str, Any]]:
    """Return all flight records."""
    return _load_json(FLIGHTS_FILE)


def load_hotels() -> List[Dict[str, Any]]:
    """Return all hotel records."""
    return _load_json(HOTELS_FILE)


def load_places() -> List[Dict[str, Any]]:
    """Return all place / attraction records."""
    return _load_json(PLACES_FILE)


def normalise_city(name: str) -> str:
    """Normalise user input such as ' goa ' or 'DELHI' to 'Goa' / 'Delhi'."""
    return (name or "").strip().title()
