import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="  [%(name)s] %(message)s",
    stream=sys.stdout,
)

from ai.agent import run_travel_agent


async def main():
    print("=== Weather Tool Binding Test ===\n")

    test_cases = [
        ("City only", "What is the current weather in Hyderabad?"),
    ]

    for label, msg in test_cases:
        print(f"[{label}]")
        print(f"User: {msg}")
        reply = await run_travel_agent(msg)
        print(f"Agent: {reply}\n{'-' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
