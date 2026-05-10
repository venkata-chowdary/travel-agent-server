"""
Real-time test for PreferenceAgent.

Usage (from the server/ directory):
    python -m ai.test_preference_agent
    python -m ai.test_preference_agent <user-uuid>

If no UUID is passed, the agent runs with a placeholder UUID — the DB calls
will fail gracefully (returning empty sections) and the LLM will still
synthesize a PreferenceContext with low memory_confidence.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# Default UUID — replace with a real one from your DB, or pass via CLI.
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


async def main(user_id: str) -> None:
    from ai.agents.preference_agent import PreferenceAgent
    from ai.tools.preference_tools import make_preference_tools

    # Run each tool individually to show raw output before any parsing.
    print("[tools]")
    tools = make_preference_tools(user_id)
    for t in tools:
        try:
            result = await t.ainvoke({})
            print(f"  {t.name}: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception:
            print(f"  {t.name}: ERROR")
            traceback.print_exc()

    print("\n[agent]")
    try:
        agent = PreferenceAgent()
        result = await agent.run(user_id)

        print(f"  status : {agent.state.get('status')}")
        if agent.state.get("error"):
            print(f"  error  : {agent.state['error']}")

        if "saved_preferences" in agent.state:
            print(f"\n  saved_preferences:")
            print(json.dumps(agent.state["saved_preferences"].model_dump(), indent=4, ensure_ascii=False))

        if "user_profile" in agent.state:
            print(f"\n  user_profile:")
            print(json.dumps(agent.state["user_profile"].model_dump(), indent=4, ensure_ascii=False))

        if "past_trips" in agent.state:
            print(f"\n  past_trips:")
            print(json.dumps([t.model_dump() for t in agent.state["past_trips"]], indent=4, ensure_ascii=False))

        print("\n  [synthesized response]")
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER_ID
    asyncio.run(main(uid))
