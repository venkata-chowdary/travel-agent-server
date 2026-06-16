"""
Integration test for the multi-agent travel planning system.

Tests the full supervisor loop:
  supervisor → preference_agent → supervisor → weather_agent → supervisor → planner

Usage (from the server/ directory):
    python -m ai.test_agents
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# ── styling ────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
RED     = "\033[91m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"

def header(title: str) -> None:
    w = 64
    print(f"\n{BOLD}{CYAN}{'═' * w}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * w}{RESET}\n")

def section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}── {title} {'─' * (54 - len(title))}{RESET}\n")

def ok(msg: str)              -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def info(label: str, val: str)-> None: print(f"  {CYAN}{label:<22}{RESET} {val}")
def warn(msg: str)            -> None: print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg: str)            -> None: print(f"  {RED}✗{RESET}  {msg}")
def dim(msg: str)             -> None: print(f"  {DIM}{msg}{RESET}")
def elapsed(t0: float)        -> str:  return f"{MAGENTA}{time.perf_counter() - t0:.2f}s{RESET}"

# ── config ─────────────────────────────────────────────────────────────────────

USER_ID = "b76a80fc-9a69-4c1d-8554-c1776c5da0d2"

QUERY_WITH_DESTINATION = (
    "Plan a 3-day trip from Hyderabad to Goa next weekend. "
    "Budget is ₹30,000 for 2 people. We love beaches and seafood."
)

QUERY_WITHOUT_DESTINATION = (
    "Give me some travel ideas for a relaxing weekend getaway."
)


# ── Test 1: PreferenceAgent standalone ────────────────────────────────────────

async def test_preference_agent() -> None:
    from ai.agents.preference_agent import build_preference_executor
    from ai.helpers import GeminiClient
    from config import settings

    section("Test 1 · PreferenceAgent (standalone)")
    print(f"  User ID : {BOLD}{USER_ID}{RESET}\n")

    llm = GeminiClient(model=settings.llm_model, temperature=0)
    agent = build_preference_executor(USER_ID, llm)
    t0 = time.perf_counter()

    result = await agent.ainvoke({
        "messages": [("human", "Fetch all user preference data and synthesize a PreferenceContext.")]
    })
    dur = elapsed(t0)

    ctx = result.get("structured_response")
    if ctx is None:
        fail(f"structured_response missing from result — keys: {list(result.keys())}")
        raise AssertionError("No structured_response")

    ok(f"PreferenceAgent done in {dur}")
    print()
    info("Home city",         ctx.home_city or "—")
    info("Travel style",      ctx.travel_style or "—")
    info("Budget style",      ctx.budget_style or "—")
    info("Food preference",   ctx.food_preference or "—")
    info("Hotel preference",  ctx.hotel_preference or "—")
    info("Currency",          ctx.currency or "—")
    info("Memory confidence", f"{ctx.memory_confidence:.0%}")
    if ctx.preferred_transport:
        info("Transport",     ", ".join(ctx.preferred_transport))
    if ctx.avoid:
        info("Avoid",         ", ".join(ctx.avoid))


# ── Test 2: WeatherAgent standalone ───────────────────────────────────────────

async def test_weather_agent(destination: str = "Goa", days: int = 3) -> None:
    from ai.agents.weather_agent import build_weather_executor
    from ai.helpers import GeminiClient
    from config import settings

    section("Test 2 · WeatherAgent (standalone)")
    trip_dates = [(date.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    print(f"  Destination : {BOLD}{destination}{RESET}")
    print(f"  Dates       : {', '.join(trip_dates)}\n")

    llm = GeminiClient(model=settings.llm_model, temperature=0)
    agent = build_weather_executor(llm)
    t0 = time.perf_counter()

    result = await agent.ainvoke({
        "messages": [("human", f"Get weather forecast for {destination} on these dates: {', '.join(trip_dates)}")]
    })
    dur = elapsed(t0)

    forecast = result.get("structured_response")
    if forecast is None:
        fail(f"structured_response missing — keys: {list(result.keys())}")
        raise AssertionError("No structured_response")

    ok(f"WeatherAgent done in {dur}")
    print()
    info("Destination",        forecast.destination)
    info("Requires replanning", str(forecast.requires_replanning))
    print()
    print(f"  {BOLD}Summary:{RESET}")
    for line in forecast.summary.split(". "):
        if line.strip():
            dim(f"  {line.strip()}.")
    print()

    if forecast.daily_forecast:
        ok(f"{len(forecast.daily_forecast)} daily forecast(s)")
        for day in forecast.daily_forecast:
            risk_color = {"low": GREEN, "medium": YELLOW, "high": RED}.get(day.risk_level, RESET)
            print(
                f"    {DIM}{day.date}{RESET}  {day.condition:<22} "
                f"{day.temperature:<14} rain {day.rain_probability:>3}%  "
                f"{risk_color}[{day.risk_level}]{RESET}"
            )
    else:
        warn("No daily forecast data")

    if forecast.trip_risks:
        print()
        ok(f"{len(forecast.trip_risks)} risk(s) flagged")
        for risk in forecast.trip_risks:
            sev_color = {"low": GREEN, "medium": YELLOW, "high": RED}.get(risk.severity, RESET)
            print(
                f"    Day {risk.day}  {sev_color}[{risk.severity}]{RESET}  "
                f"{risk.risk_type}: {risk.recommendation}"
            )
    else:
        ok("No trip risks flagged")


# ── Test 3: Full pipeline WITH destination ─────────────────────────────────────

async def test_full_pipeline_with_destination() -> None:
    from ai.agent import run_travel_agent

    section("Test 3 · Full Pipeline — with destination (supervisor should route to both agents)")
    print(f"  User ID : {BOLD}{USER_ID}{RESET}")
    print(f"  Query   : {BOLD}{QUERY_WITH_DESTINATION}{RESET}\n")

    t0 = time.perf_counter()
    response = await run_travel_agent(user_id=USER_ID, user_message=QUERY_WITH_DESTINATION)
    dur = elapsed(t0)

    ok(f"Pipeline completed in {dur}")
    print()
    info("Destination",  response.destination)
    info("Origin",       response.origin or "—")
    info("Duration",     f"{response.days} day(s)")
    info("Travelers",    str(response.travelers))
    print()

    print(f"  {BOLD}Summary:{RESET}")
    for line in response.summary.split(". "):
        if line.strip():
            dim(f"  {line.strip()}.")
    print()

    b = response.budget
    ok(f"Budget ({b.currency})")
    print(f"    Flights    {b.currency}{b.flights:,.0f}")
    print(f"    Stay       {b.currency}{b.stay:,.0f}")
    print(f"    Activities {b.currency}{b.activities:,.0f}")
    print(f"    Food       {b.currency}{b.food:,.0f}")
    print(f"    {'─' * 22}")
    print(f"    Total      {BOLD}{b.currency}{b.total:,.0f}{RESET}")
    print()

    if response.itinerary:
        ok(f"Itinerary: {len(response.itinerary)} day(s)")
        for day in response.itinerary:
            print(f"\n    {BOLD}Day {day.day}{RESET} — {day.title}")
            for item in day.items[:3]:
                print(f"      {DIM}{item.time:<8}{RESET} [{item.type}] {item.title}")
            if len(day.items) > 3:
                dim(f"      … +{len(day.items) - 3} more")
    else:
        warn("No itinerary in response")
    print()

    if response.daily_forecast:
        ok(f"Weather embedded: {len(response.daily_forecast)} day(s)")
        if response.weather_summary:
            dim(f"  {response.weather_summary[:120]}")
        if response.requires_replanning:
            warn("Requires replanning due to weather")
    else:
        warn("No weather data — supervisor may have skipped WeatherAgent")

    if response.verification_tips:
        print()
        ok(f"{len(response.verification_tips)} verification tip(s)")
        for tip in response.verification_tips[:3]:
            dim(f"  • {tip}")


# ── Test 4: Full pipeline WITHOUT destination ──────────────────────────────────

async def test_full_pipeline_no_destination() -> None:
    from ai.agent import run_travel_agent

    section("Test 4 · Full Pipeline — no destination (supervisor should skip WeatherAgent)")
    print(f"  User ID : {BOLD}{USER_ID}{RESET}")
    print(f"  Query   : {BOLD}{QUERY_WITHOUT_DESTINATION}{RESET}\n")

    t0 = time.perf_counter()
    response = await run_travel_agent(user_id=USER_ID, user_message=QUERY_WITHOUT_DESTINATION)
    dur = elapsed(t0)

    ok(f"Pipeline completed in {dur}")
    print()
    info("Destination",  response.destination or "—")
    info("Duration",     f"{response.days} day(s)")
    print()
    print(f"  {BOLD}Summary:{RESET}")
    for line in response.summary.split(". "):
        if line.strip():
            dim(f"  {line.strip()}.")
    print()

    if response.daily_forecast:
        warn("Weather data present — unexpected for a no-destination query")
    else:
        ok("WeatherAgent correctly skipped (no destination)")


# ── main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    header("Travel Agent · Multi-Agent System Test")
    print(f"  {DIM}Started  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}{RESET}")
    print(f"  {DIM}User ID  : {USER_ID}{RESET}")

    results: dict[str, str] = {}

    for name, coro in [
        ("PreferenceAgent (standalone)",      test_preference_agent()),
        ("WeatherAgent (standalone)",         test_weather_agent()),
        ("Full pipeline — with destination",  test_full_pipeline_with_destination()),
        ("Full pipeline — no destination",    test_full_pipeline_no_destination()),
    ]:
        try:
            await coro
            results[name] = f"{GREEN}PASS{RESET}"
        except Exception as exc:
            results[name] = f"{RED}FAIL{RESET}"
            fail(f"{name} → {exc}")
            import traceback; traceback.print_exc()

    section("Results")
    for name, status in results.items():
        print(f"  {status}  {name}")

    print()
    if all("PASS" in s for s in results.values()):
        print(f"  {BOLD}{GREEN}All tests passed.{RESET}\n")
    else:
        print(f"  {BOLD}{RED}Some tests failed — check output above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
