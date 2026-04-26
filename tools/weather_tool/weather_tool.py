from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.request import urlopen

from langchain_core.tools import tool


def _http_get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


@tool
def get_current_date() -> str:
    """Return the current UTC date and weekday for planning context."""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%d')} ({now.strftime('%A')}) UTC"


@tool
def get_current_weather(city: str, country_code: str | None = None) -> str:
    """
    Fetch current weather for a location using wttr.in.

    Args:
        city: City name (for example: "Bali", "Tokyo", "Hyderabad").
        country_code: Optional ISO country code like "IN" or "JP".
    """
    location = city.strip()
    if country_code:
        location = f"{location},{country_code.strip().upper()}"

    weather_url = f"https://wttr.in/{location}?format=j1"
    weather_data = _http_get_json(weather_url)
    current_list = weather_data.get("current_condition") or []
    nearest_list = weather_data.get("nearest_area") or []

    if not current_list:
        return f"Weather unavailable for {city}."

    current = current_list[0]
    nearest = nearest_list[0] if nearest_list else {}
    area = (nearest.get("areaName") or [{}])[0].get("value", city)
    country = (nearest.get("country") or [{}])[0].get("value", "")
    condition = (current.get("weatherDesc") or [{}])[0].get("value", "Unknown")

    return (
        f"{area}, {country}: {condition}, "
        f"{current.get('temp_C')} degC (feels {current.get('FeelsLikeC')} degC), "
        f"humidity {current.get('humidity')}%, "
        f"wind {current.get('windspeedKmph')} km/h."
    )
