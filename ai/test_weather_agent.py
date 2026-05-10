"""
Real-time test for WeatherAgent.

Usage (from the server/ directory):
    python -m ai.test_weather_agent
    python -m ai.test_weather_agent <destination> <date1> [<date2> ...]

Examples:
    python -m ai.test_weather_agent
    python -m ai.test_weather_agent "Mumbai" "2026-05-12" "2026-05-13" "2026-05-14"
    python -m ai.test_weather_agent "Manali" "2026-06-01" "2026-06-02" "2026-06-03"
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# Default test cases run when no CLI args are provided.
DEFAULT_CASES: list[tuple[str, list[str]]] = [
    ("Goa",     ["2026-05-15", "2026-05-16", "2026-05-17"]),
    ("Manali",  ["2026-06-01", "2026-06-02", "2026-06-03"]),
    ("Mumbai",  ["2026-05-12"]),
]


async def run_case(destination: str, trip_dates: list[str]) -> None:
    from ai.agents.weather_agent import WeatherAgent
    from ai.tools.weather_tool import get_weather_forecast

    print(f"[tool]  destination={destination!r}  dates={trip_dates}")
    try:
        raw = get_weather_forecast.invoke({"city": destination, "trip_dates": trip_dates})
        print(json.dumps(raw, indent=2, ensure_ascii=False))
    except Exception:
        print("  ERROR")
        traceback.print_exc()

    print("\n[llm synthesis]")
    try:
        agent = WeatherAgent()
        result = await agent.run(destination, trip_dates)

        print(f"  status : {agent.state.get('status')}")
        if agent.state.get("error"):
            print(f"  error  : {agent.state['error']}")

        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    except Exception:
        traceback.print_exc()


async def main() -> None:
    args = sys.argv[1:]

    if args:
        destination = args[0]
        trip_dates = args[1:] if len(args) > 1 else ["2026-05-15"]
        await run_case(destination, trip_dates)
    else:
        for destination, trip_dates in DEFAULT_CASES:
            await run_case(destination, trip_dates)
            print()


if __name__ == "__main__":
    asyncio.run(main())
