from __future__ import annotations

import logging

from langgraph.prebuilt import create_react_agent

from ai.helpers import get_llm
from ai.prompts import PREFERENCE_AGENT_SYSTEM_PROMPT
from ai.schemas.preferences import PreferenceContext
from ai.state import TravelState, set_status
from ai.tools.preference_tools import make_preference_tools
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


async def preference_agent_node(state: TravelState) -> dict:
    logger.info("PreferenceAgent running for user %s", state["user_id"])

    agent = create_react_agent(
        _llm,
        make_preference_tools(state["user_id"]),
        prompt=PREFERENCE_AGENT_SYSTEM_PROMPT,
        response_format=PreferenceContext,
    )

    try:
        result = await agent.ainvoke({
            "messages": [("human", "Fetch all user preference data and synthesize a PreferenceContext.")]
        })
        ctx: PreferenceContext = result["structured_response"]
        has_data = bool(
            ctx.travel_style or ctx.budget_style or ctx.preferred_transport
            or ctx.food_preference or ctx.hotel_preference or ctx.avoid
            or ctx.origin or ctx.memory_confidence > 0
        )
        status = "succeeded" if has_data else "empty"
        logger.info("PreferenceAgent done — origin: %s, budget: %s", ctx.origin, ctx.budget_style)
        return {
            "preference_context": ctx,
            "workflow_statuses": set_status(state, "preferences", status),
        }
    except Exception:
        logger.error("PreferenceAgent failed", exc_info=True)
        return {
            "preference_context": PreferenceContext(
                supervisor_note="Preference agent failed — no prior context available. Clarification may help.",
            ),
            "workflow_statuses": set_status(state, "preferences", "failed"),
        }
