"""
End-to-end test for the PreferenceAgent pipeline.

Run from the server/ directory:
    python ai/test_preference_agent.py

Flow:
  1. Runs PreferenceAgent.run() — real tool calls + real LLM synthesis
  2. Prints the synthesized PreferenceContext
  3. Runs run_travel_agent() with the context — real main agent response
"""

import asyncio
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="  [%(name)s] %(message)s",
    stream=sys.stdout,
)

from db import init_db
from ai.agents import PreferenceAgent
from ai.agent import run_travel_agent
from ai.schemas import PreferenceContext

EXISTING_USER_ID = uuid.UUID("fa4db719-5f1c-47ea-9246-55073e995c11")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


async def run_preference_agent_test(user_id: uuid.UUID) -> PreferenceContext:
    section("Step 1 — PreferenceAgent.run()")
    print("  Calling tools: get_saved_preferences, get_user_profile, get_past_trips")
    print("  Synthesizing PreferenceContext via LLM...\n")

    agent = PreferenceAgent()
    ctx = await agent.run(user_id)

    print(f"  travel_style       : {ctx.travel_style}")
    print(f"  budget_style       : {ctx.budget_style}")
    print(f"  preferred_transport: {ctx.preferred_transport}")
    print(f"  food_preference    : {ctx.food_preference}")
    print(f"  hotel_preference   : {ctx.hotel_preference}")
    print(f"  avoid              : {ctx.avoid}")
    print(f"  home_city          : {ctx.home_city}")
    print(f"  currency           : {ctx.currency}")
    print(f"  memory_confidence  : {ctx.memory_confidence:.2f}")

    assert isinstance(ctx, PreferenceContext), "Expected PreferenceContext"
    assert 0.0 <= ctx.memory_confidence <= 1.0, "memory_confidence out of bounds"
    assert ctx.home_city is not None, "home_city should be set — check DB preferences"
    assert ctx.currency == "₹", f"currency should be ₹, got {ctx.currency!r}"
    print("\n  ✓ Valid PreferenceContext returned")
    return ctx


async def run_main_agent_test(ctx: PreferenceContext) -> None:
    section("Step 2 — run_travel_agent() with PreferenceContext")

    message = "Plan a 3-day weekend trip for this weekend. Keep it budget-friendly and relaxed."
    print(f"  Message: {message}\n")

    reply = await run_travel_agent(message, preference_context=ctx)

    print("  Agent reply:\n")
    for line in reply.strip().splitlines():
        print(f"    {line}")

    assert isinstance(reply, str) and len(reply) > 0
    print("\n  ✓ Main agent returned a valid response")


async def main():
    print("\n=== Preference Agent End-to-End Test ===")

    await init_db()
    print(f"\n  Using existing user: {EXISTING_USER_ID}")

    ctx = await run_preference_agent_test(EXISTING_USER_ID)
    await run_main_agent_test(ctx)

    section("Done — all steps passed")


if __name__ == "__main__":
    asyncio.run(main())
