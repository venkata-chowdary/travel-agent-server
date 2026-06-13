"""
Real-time integration test for the multi-agent travel planning system.

Exercises the full collaboration chain:
  PreferenceAgent  →  WeatherAgent  →  Main LangGraph agent

Usage (from the server/ directory):
    python -m ai.test_agents
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# ─── styling helpers ─────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"

def header(title: str) -> None:
    width = 64
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")

def section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}── {title} {'─' * (54 - len(title))}{RESET}\n")

def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")

def info(label: str, value: str) -> None:
    print(f"  {CYAN}{label:<22}{RESET} {value}")

def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")

def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")

def dim(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")

def elapsed(t0: float) -> str:
    return f"{MAGENTA}{time.perf_counter() - t0:.2f}s{RESET}"


# ─── test config ─────────────────────────────────────────────────────────────

USER_ID      = "fa4db719-5f1c-47ea-9246-55073e995c11"
TRAVEL_QUERY = (
    "Plan a 5-day trip to Goa for 2 people in mid-July. "
    "We love beaches, seafood, and local nightlife. "
    "Total budget is ₹60,000 including flights from Hyderabad."
)


# ─── individual agent tests ───────────────────────────────────────────────────

async def test_preference_agent() -> None:
    from ai.agents.preference_agent import PreferenceAgent

    section("Test 1 · PreferenceAgent")
    print(f"  User ID  : {BOLD}{USER_ID}{RESET}")
    print()

    agent = PreferenceAgent()
    t0 = time.perf_counter()

    try:
        ctx = await agent.run(user_id=USER_ID)
        dur = elapsed(t0)

        ok(f"PreferenceAgent finished in {dur}")
        print()
        info("Status",            agent.state.get("status", "—"))
        info("Home city",         ctx.home_city or "—")
        info("Travel style",      ctx.travel_style or "—")
        info("Budget style",      ctx.budget_style or "—")
        info("Food preference",   ctx.food_preference or "—")
        info("Hotel preference",  ctx.hotel_preference or "—")
        info("Currency",          ctx.currency or "—")
        info("Memory confidence", f"{ctx.memory_confidence:.0%}" if ctx.memory_confidence else "—")

        if ctx.preferred_transport:
            info("Transport",     ", ".join(ctx.preferred_transport))
        if ctx.avoid:
            info("Avoid",         ", ".join(ctx.avoid))

        print()
        if "saved_preferences" in agent.state:
            ok("saved_preferences tool returned data")
        else:
            warn("saved_preferences returned nothing")

        if "user_profile" in agent.state:
            p = agent.state["user_profile"]
            ok(f"user_profile → name={p.name!r}, email={p.email!r}")
        else:
            warn("user_profile returned nothing")

        if "past_trips" in agent.state:
            n = len(agent.state["past_trips"])
            ok(f"past_trips → {n} trip(s) found")
        else:
            warn("past_trips returned nothing")

    except Exception as exc:
        fail(f"PreferenceAgent raised: {exc}")
        raise

    return ctx          # pass forward


async def test_weather_agent(destination: str = "Goa") -> None:
    from ai.agents.weather_agent import WeatherAgent
    from datetime import date, timedelta

    section("Test 2 · WeatherAgent")
    today = date.today()
    trip_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    print(f"  Destination : {BOLD}{destination}{RESET}")
    print(f"  Trip dates  : {', '.join(trip_dates)}")
    print()

    agent = WeatherAgent()
    t0 = time.perf_counter()

    try:
        forecast = await agent.run(destination=destination, trip_dates=trip_dates)
        dur = elapsed(t0)

        ok(f"WeatherAgent finished in {dur}")
        print()
        info("Status",             agent.state.get("status", "—"))
        info("Destination",        forecast.destination)
        info("Requires replanning",str(forecast.requires_replanning))
        print()
        print(f"  {BOLD}Summary:{RESET}")
        for line in forecast.summary.split(". "):
            if line.strip():
                dim(f"  {line.strip()}.")
        print()

        if forecast.daily_forecast:
            ok(f"{len(forecast.daily_forecast)} daily forecast(s) received")
            for day in forecast.daily_forecast:
                risk_color = {
                    "low": GREEN, "medium": YELLOW, "high": RED
                }.get(day.risk_level, RESET)
                print(
                    f"    {DIM}{day.date}{RESET}  "
                    f"{day.condition:<22} {day.temperature:<14} "
                    f"rain {day.rain_probability:>3}%  "
                    f"{risk_color}[{day.risk_level}]{RESET}"
                )
        else:
            warn("No daily forecast data returned")

        if forecast.trip_risks:
            print()
            ok(f"{len(forecast.trip_risks)} trip risk(s) detected")
            for risk in forecast.trip_risks:
                sev_color = {
                    "low": GREEN, "medium": YELLOW, "high": RED
                }.get(risk.severity, RESET)
                print(
                    f"    Day {risk.day}  "
                    f"{sev_color}[{risk.severity}]{RESET}  "
                    f"{risk.risk_type}: {risk.recommendation}"
                )
        else:
            ok("No trip risks flagged")

    except Exception as exc:
        fail(f"WeatherAgent raised: {exc}")
        raise

    return forecast


async def test_full_pipeline() -> None:
    from ai.agent import run_travel_agent

    section("Test 3 · Full Multi-Agent Pipeline (main graph)")
    print(f"  User ID  : {BOLD}{USER_ID}{RESET}")
    print(f"  Query    : {BOLD}{TRAVEL_QUERY}{RESET}")
    print()

    t0 = time.perf_counter()

    try:
        response = await run_travel_agent(
            user_id=USER_ID,
            user_message=TRAVEL_QUERY,
        )
        dur = elapsed(t0)

        ok(f"Full pipeline completed in {dur}")
        print()

        # ── Trip overview ────────────────────────────────────────────────────
        info("Trip ID",          response.id)
        info("Destination",      response.destination)
        info("Origin",           response.origin or "—")
        info("Duration",         f"{response.days} days")
        info("Travelers",        str(response.travelers))
        info("Status",           response.status)
        info("Created at",       response.created_at)
        print()

        # ── Summary ──────────────────────────────────────────────────────────
        print(f"  {BOLD}Trip summary:{RESET}")
        for line in response.summary.split(". "):
            if line.strip():
                dim(f"  {line.strip()}.")
        print()

        # ── Budget ───────────────────────────────────────────────────────────
        b = response.budget
        ok(f"Budget breakdown ({b.currency})")
        print(f"    Flights      {b.currency}{b.flights:,.0f}")
        print(f"    Stay         {b.currency}{b.stay:,.0f}")
        print(f"    Activities   {b.currency}{b.activities:,.0f}")
        print(f"    Food         {b.currency}{b.food:,.0f}")
        print(f"    {'─' * 24}")
        print(f"    Total        {BOLD}{b.currency}{b.total:,.0f}{RESET}")
        print()

        # ── Itinerary ────────────────────────────────────────────────────────
        if response.itinerary:
            ok(f"Itinerary: {len(response.itinerary)} day(s)")
            for day in response.itinerary:
                print(f"\n    {BOLD}Day {day.day}{RESET} — {day.title}")
                for item in day.items[:4]:   # cap at 4 items to keep output tidy
                    print(f"      {DIM}{item.time:<8}{RESET} {item.title}")
                if len(day.items) > 4:
                    dim(f"      … +{len(day.items) - 4} more items")
        else:
            warn("No itinerary returned")
        print()

        # ── Weather passthrough ──────────────────────────────────────────────
        if response.daily_forecast:
            ok(f"Weather embedded: {len(response.daily_forecast)} day(s)")
            if response.weather_summary:
                dim(f"  {response.weather_summary[:120]}…")
            if response.requires_replanning:
                warn("Requires replanning due to severe weather")
        else:
            warn("No weather data embedded (destination may not have been detected)")

        # ── Verification tips ────────────────────────────────────────────────
        if response.verification_tips:
            print()
            ok(f"{len(response.verification_tips)} verification tip(s)")
            for tip in response.verification_tips[:3]:
                dim(f"  • {tip}")

        # ── Raw JSON snapshot ────────────────────────────────────────────────
        print()
        print(f"  {DIM}Raw JSON (first 800 chars):{RESET}")
        raw = response.model_dump_json(indent=2)
        print(f"  {DIM}{raw[:800]}…{RESET}" if len(raw) > 800 else f"  {DIM}{raw}{RESET}")

    except Exception as exc:
        fail(f"Full pipeline raised: {exc}")
        import traceback
        traceback.print_exc()
        raise


# ─── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    header("Travel Agent · Multi-Agent Collaboration Test")
    print(f"  {DIM}Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}{RESET}")
    print(f"  {DIM}User ID   : {USER_ID}{RESET}")

    results: dict[str, str] = {}

    # ── 1. Preference agent ──────────────────────────────────────────────────
    try:
        await test_preference_agent()
        results["PreferenceAgent"] = f"{GREEN}PASS{RESET}"
    except Exception:
        results["PreferenceAgent"] = f"{RED}FAIL{RESET}"

    # ── 2. Weather agent ─────────────────────────────────────────────────────
    try:
        await test_weather_agent(destination="Goa")
        results["WeatherAgent"] = f"{GREEN}PASS{RESET}"
    except Exception:
        results["WeatherAgent"] = f"{RED}FAIL{RESET}"

    # ── 3. Full pipeline (both sub-agents + main LLM) ────────────────────────
    try:
        await test_full_pipeline()
        results["Full pipeline"] = f"{GREEN}PASS{RESET}"
    except Exception:
        results["Full pipeline"] = f"{RED}FAIL{RESET}"

    # ── Summary ──────────────────────────────────────────────────────────────
    section("Results")
    for name, status in results.items():
        print(f"  {status}  {name}")

    all_pass = all("PASS" in s for s in results.values())
    print()
    if all_pass:
        print(f"  {BOLD}{GREEN}All tests passed.{RESET}\n")
    else:
        print(f"  {BOLD}{RED}Some tests failed — check output above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
