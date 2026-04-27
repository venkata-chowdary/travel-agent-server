from __future__ import annotations

import argparse
import asyncio
import json
import sys
from uuid import uuid4

try:
    from ai.agent import run_travel_agent
except ModuleNotFoundError:
    # Allow running this module directly from `server/ai`.
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from agent import run_travel_agent


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Quick local runner for the travel agent.")
    parser.add_argument(
        "--message",
        default="Plan a 3-day budget trip to Goa for 2 travelers from Hyderabad.",
        help="User message to send to the travel agent.",
    )
    parser.add_argument(
        "--user-id",
        default=str(uuid4()),
        help="User ID used to fetch stored preferences.",
    )
    args = parser.parse_args()

    response = await run_travel_agent(
        user_id=args.user_id,
        user_message=args.message,
        history=None,
    )

    # Supports both Pydantic v1 and v2 model serialization.
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    else:
        payload = response.dict()

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
