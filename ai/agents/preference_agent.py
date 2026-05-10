from __future__ import annotations

import json
import sys
from typing import TypedDict
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ai.helpers import GeminiClient
from ai.prompts import PREFERENCE_AGENT_SYSTEM_PROMPT
from ai.schemas import PreferenceContext, TravelPreferences
from ai.schemas.preferences import PastTrip, UserProfile
from ai.tools.preference_tools import make_preference_tools
from config import settings

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


class PreferenceAgentState(TypedDict, total=False):
    status: str                          # "idle" | "fetched" | "synthesized" | "error"
    user_id: str
    saved_preferences: TravelPreferences
    user_profile: UserProfile
    past_trips: list[PastTrip]
    response: PreferenceContext
    error: str


class PreferenceAgent:
    """
    Gathers user data by calling preference tools directly (no LLM routing),
    then synthesizes a PreferenceContext via a single structured-output LLM call.

    Avoids LangGraph to prevent conflicts with the Google GenAI SDK's
    Automatic Function Calling (AFC), which would deadlock with a tool-calling loop.

    Execution state is tracked in self.state (PreferenceAgentState) throughout run().
    Each tool's raw output is parsed into a typed Pydantic model before being stored.
    """

    def __init__(self) -> None:
        self._llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)
        self.state: PreferenceAgentState = {"status": "idle"}

    async def run(self, user_id: str | UUID) -> PreferenceContext:
        self.state = {"status": "idle", "user_id": str(user_id)}

        try:
            tools = make_preference_tools(user_id)
            sections: list[str] = []

            for t in tools:
                try:
                    raw = await t.ainvoke({})

                    if t.name == "get_saved_preferences":
                        parsed = TravelPreferences.model_validate(raw)
                        self.state["saved_preferences"] = parsed
                        sections.append(f"[saved_preferences]\n{json.dumps(parsed.model_dump(), indent=2)}")

                    elif t.name == "get_user_profile":
                        parsed = UserProfile.model_validate(raw)
                        self.state["user_profile"] = parsed
                        sections.append(f"[user_profile]\n{json.dumps(parsed.model_dump(), indent=2)}")

                    elif t.name == "get_past_trips":
                        parsed = [PastTrip.model_validate(trip) for trip in raw]
                        self.state["past_trips"] = parsed
                        sections.append(f"[past_trips]\n{json.dumps([p.model_dump() for p in parsed], indent=2)}")

                except Exception:
                    sections.append(f"[{t.name}]\n(unavailable)")

            self.state["status"] = "fetched"
            tool_data = "\n\n".join(sections)

            result = await self._llm.with_structured_output(PreferenceContext, method="json_schema").ainvoke([
                HumanMessage(content=(
                    f"{PREFERENCE_AGENT_SYSTEM_PROMPT}\n\n"
                    f"Raw data:\n\n{tool_data}\n\n"
                    "Produce the PreferenceContext JSON."
                ))
            ])
            self.state["response"] = result
            self.state["status"] = "synthesized"
            return result

        except Exception:
            self.state["status"] = "error"
            self.state["error"] = "Preference synthesis failed."
            return PreferenceContext()
