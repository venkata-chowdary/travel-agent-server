from __future__ import annotations

import asyncio
import json
from statistics import mode
from urllib.request import urlopen

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherToolOutput(BaseModel):
    city: str = Field(description="Resolved city name from the weather provider.")
    country: str = Field(description="Country name.")
    condition: str = Field(description="Human-readable weather condition, e.g. 'Partly Cloudy'.")
    temp_c: float = Field(description="Current temperature in Celsius.")
    feels_like_c: float = Field(description="Feels-like temperature in Celsius.")
    humidity_pct: int = Field(description="Relative humidity as a percentage (0â€“100).")
    wind_kmph: int = Field(description="Wind speed in km/h.")
    summary: str = Field(description="One-sentence plain-English summary suitable for travel advice.")


def _sync_http_get(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


async def _http_get_json(url: str, retries: int = 3) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(_sync_http_get, url)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # 1s then 2s before retry
    raise last_exc  # type: ignore[misc]


@tool
async def get_current_weather(city: str) -> dict:
    """
    Fetch current weather for a travel destination using wttr.in.

    Pass the full, unambiguous city name as wttr.in uses it for geolocation â€”
    Do NOT append country codes (e.g. pass "Gokarna" not "Gokarna,IN").
    For internationally ambiguous names use the region or state to disambiguate
    (e.g. "Hospet Karnataka" or "Springfield Illinois").
    Only call this once per planning response â€” pick the primary destination.

    Args:
        city: Full city name, e.g. "Gokarna", "Hyderabad", "Hospet Karnataka".

    Returns:
        WeatherToolOutput fields: city, country, condition, temp_c, feels_like_c,
        humidity_pct, wind_kmph, summary.
    """
    location = city.strip().replace(" ", "+")
    weather_url = f"https://wttr.in/{location}?format=j1"

    try:
        weather_data = await _http_get_json(weather_url)
    except Exception:
        return WeatherToolOutput(
            city=city, country="", condition="Unavailable",
            temp_c=0.0, feels_like_c=0.0, humidity_pct=0, wind_kmph=0,
            summary=f"Weather data is currently unavailable for {city}.",
        ).model_dump()

    current_list = weather_data.get("current_condition") or []
    nearest_list = weather_data.get("nearest_area") or []

    if not current_list:
        return WeatherToolOutput(
            city=city, country="", condition="Unavailable",
            temp_c=0.0, feels_like_c=0.0, humidity_pct=0, wind_kmph=0,
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
        f"It is currently {condition.lower()} in {area}, {country} with a temperature of "
        f"{temp_c}Â°C (feels like {feels_like_c}Â°C), "
        f"{humidity_pct}% humidity, and winds at {wind_kmph} km/h."
    )

    return WeatherToolOutput(
        city=area, country=country, condition=condition,
        temp_c=temp_c, feels_like_c=feels_like_c,
        humidity_pct=humidity_pct, wind_kmph=wind_kmph,
        summary=summary,
    ).model_dump()


@tool
async def get_weather_forecast(city: str, trip_dates: list[str]) -> dict:
    """
    Fetch a multi-day weather forecast for a travel destination using wttr.in.

    Returns forecast data for up to 3 days. For trips longer than 3 days the
    response includes forecast_limited=true so callers can note the limitation.

    Args:
        city: Full city name, e.g. "Goa", "Hyderabad", "Manali".
        trip_dates: List of date strings in "YYYY-MM-DD" format, one per trip day.

    Returns:
        dict with keys: city, days (list of day forecasts), forecast_limited (bool).
        Each day has: date, max_temp_c, min_temp_c, dominant_condition, max_rain_pct.
    """
    location = city.strip().replace(" ", "+")
    url = f"https://wttr.in/{location}?format=j1"

    try:
        data = await _http_get_json(url)
    except Exception:
        return {
            "city": city,
            "days": [],
            "forecast_limited": False,
            "error": f"Weather data unavailable for {city}.",
        }

    weather_days = data.get("weather") or []
    nearest_list = data.get("nearest_area") or []

    # wttr.in silently returns empty weather[] for unrecognised city names.
    # Retry once with "+India" appended â€” covers most Indian cities.
    if not weather_days:
        try:
            fallback = await _http_get_json(f"https://wttr.in/{location}+India?format=j1")
            weather_days = fallback.get("weather") or []
            if weather_days:
                nearest_list = fallback.get("nearest_area") or nearest_list
        except Exception:
            pass

    if not weather_days:
        return {
            "city": city,
            "days": [],
            "forecast_limited": False,
            "error": (
                f"No forecast data found for '{city}'. "
                "Try a nearby larger city or add a state/region name (e.g. 'Gokarna Karnataka')."
            ),
        }

    resolved_city = (nearest_list[0].get("areaName") or [{}])[0].get("value", city) if nearest_list else city

    days_out = []
    for i, trip_date in enumerate(trip_dates):
        if i >= len(weather_days):
            break
        day_data = weather_days[i]
        hourly = day_data.get("hourly") or []

        max_temp = int(day_data.get("maxtempC", 0))
        min_temp = int(day_data.get("mintempC", 0))

        conditions = [
            (h.get("weatherDesc") or [{}])[0].get("value", "")
            for h in hourly if h.get("weatherDesc")
        ]
        try:
            dominant_condition = mode(conditions) if conditions else "Unknown"
        except Exception:
            dominant_condition = conditions[0] if conditions else "Unknown"

        rain_pcts = [int(h.get("chanceofrain", 0)) for h in hourly]
        max_rain = max(rain_pcts) if rain_pcts else 0

        days_out.append({
            "date": trip_date,
            "max_temp_c": max_temp,
            "min_temp_c": min_temp,
            "dominant_condition": dominant_condition,
            "max_rain_pct": max_rain,
        })

    return {
        "city": resolved_city,
        "days": days_out,
        "forecast_limited": len(trip_dates) > len(weather_days),
    }
