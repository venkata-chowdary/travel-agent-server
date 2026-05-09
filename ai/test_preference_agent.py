"""
End-to-end test for the PreferenceAgent pipeline.

Run from the server/ directory:
    python ai/test_preference_agent.py

Flow:
  1. Creates a real test user in the DB with rich travel preferences
  2. Runs PreferenceAgent.run() — real tool calls + real LLM synthesis
  3. Prints the synthesized PreferenceContext
  4. Runs run_travel_agent() with the context — real main agent response
  5. Cleans up the test user from the DB
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

from sqlalchemy import delete

from auth.models import User
from auth.security import hash_password
from db import SessionLocal, init_db
from ai.agents import PreferenceAgent
from ai.agent import run_travel_agent
from ai.schemas import PreferenceContext

TEST_USER = {
    "email": f"test_pref_{uuid.uuid4().hex[:8]}@test.local",
    "name": "Arjun Reddy",
    "password_hash": hash_password("TestPass123"),
    "preferences": {
        "budget_range": "budget",
        "travel_style": ["relaxation", "cultural"],
        "dietary_restrictions": ["vegetarian"],
        "cabin_class": "economy",
        "accommodation_type": "hostel",
        "pace": "relaxed",
        "home_city": "Hyderabad",
        "currency": "INR",
    },
}


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


async def create_test_user() -> uuid.UUID:
    section("Setup — creating test user in DB")
    async with SessionLocal() as session:
        user = User(
            email=TEST_USER["email"],
            name=TEST_USER["name"],
            password_hash=TEST_USER["password_hash"],
            preferences=TEST_USER["preferences"],
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        user_id = user.id

    print(f"  Email      : {TEST_USER['email']}")
    print(f"  Name       : {TEST_USER['name']}")
    print(f"  User ID    : {user_id}")
    print(f"  Preferences: {TEST_USER['preferences']}")
    return user_id


async def delete_test_user(user_id: uuid.UUID) -> None:
    section("Teardown — removing test user")
    async with SessionLocal() as session:
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()
    print(f"  Deleted user {user_id}")


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
    print("\n  ✓ Valid PreferenceContext returned")
    return ctx


async def run_main_agent_test(ctx: PreferenceContext) -> None:
    section("Step 2 — run_travel_agent() with PreferenceContext")

    message = "Plan a 3-day weekend trip from Hyderabad. Keep it budget-friendly and relaxed."
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
    user_id = await create_test_user()

    try:
        ctx = await run_preference_agent_test(user_id)
        await run_main_agent_test(ctx)
    finally:
        await delete_test_user(user_id)

    section("Done — all steps passed")


if __name__ == "__main__":
    asyncio.run(main())
