from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.request import urlopen

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherToolOutput(BaseModel):
    city: str = Field(description="Resolved city name from the weather provider.")
    country: str = Field(description="Country name.")
    condition: str = Field(description="Human-readable weather condition, e.g. 'Partly Cloudy'.")
    temp_c: float = Field(description="Current temperature in Celsius.")
    feels_like_c: float = Field(description="Feels-like temperature in Celsius.")
    humidity_pct: int = Field(description="Relative humidity as a percentage (0–100).")
    wind_kmph: int = Field(description="Wind speed in km/h.")
    summary: str = Field(description="One-sentence plain-English summary suitable for travel advice.")


def _http_get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


@tool
def get_current_date() -> str:
    """Return the current UTC date and weekday for planning context."""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%d')} ({now.strftime('%A')}) UTC"


@tool
def get_current_weather(city: str, country_code: str | None = None) -> dict:
    """
    Fetch current weather for a location using wttr.in.

    Args:
        city: City name (for example: "Bali", "Tokyo", "Hyderabad").
        country_code: Optional ISO country code like "IN" or "JP".

    Returns:
        WeatherToolOutput fields: city, country, condition, temp_c, feels_like_c,
        humidity_pct, wind_kmph, summary.
    """
    location = city.strip()
    if country_code:
        location = f"{location},{country_code.strip().upper()}"

    weather_url = f"https://wttr.in/{location}?format=j1"
    weather_data = _http_get_json(weather_url)
    current_list = weather_data.get("current_condition") or []
    nearest_list = weather_data.get("nearest_area") or []

    if not current_list:
        return WeatherToolOutput(
            city=city,
            country="",
            condition="Unavailable",
            temp_c=0.0,
            feels_like_c=0.0,
            humidity_pct=0,
            wind_kmph=0,
            summary=f"Weather data is currently unavailable for {city}.",
        ).model_dump()

    current = current_list[0]
    nearest = nearest_list[0] if nearest_list else {}
    area = (nearest.get("areaName") or [{}])[0].get("value", city)
    country = (nearest.get("country") or [{}])[0].get("value", "")
    condition = (current.get("weatherDesc") or [{}])[0].get("value", "Unknown")
    temp_c = float(current.get("temp_C", 0))
    feels_like_c = float(current.get("FeelsLikeC", 0))
    humidity_pct = int(current.get("humidity", 0))
    wind_kmph = int(current.get("windspeedKmph", 0))

    summary = (
        f"It is currently {condition.lower()} in {area} with a temperature of "
        f"{temp_c}°C (feels like {feels_like_c}°C), "
        f"{humidity_pct}% humidity, and winds at {wind_kmph} km/h."
    )

    return WeatherToolOutput(
        city=area,
        country=country,
        condition=condition,
        temp_c=temp_c,
        feels_like_c=feels_like_c,
        humidity_pct=humidity_pct,
        wind_kmph=wind_kmph,
        summary=summary,
    ).model_dump()

