import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.agent import run_travel_agent


async def main():
    print("=== Travel Agent Test ===\n")

    test_messages = [
        "Plan a 3-day trip to Paris for 2 people.",
        "What should I pack for a beach trip to Goa in July?",
        "Give me a budget breakdown for a week in Tokyo.",
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        reply = await run_travel_agent(msg)
        print(f"Agent: {reply}\n{'-' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
