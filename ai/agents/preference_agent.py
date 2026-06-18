from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ai.helpers import GeminiClient
from ai.prompts import PREFERENCE_AGENT_SYSTEM_PROMPT
from ai.schemas.preferences import PreferenceContext
from ai.state import TravelState, _preference_has_data, _status_update
from ai.tools.preference_tools import make_preference_tools
from config import settings

logger = logging.getLogger(__name__)
_llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)


def build_preference_executor(user_id: str, llm: BaseChatModel):
    return create_agent(
        model=llm,
        tools=make_preference_tools(user_id),
        system_prompt=PREFERENCE_AGENT_SYSTEM_PROMPT,
        response_format=PreferenceContext,
    )


async def preference_agent_node(state: TravelState) -> dict:
    logger.info("PreferenceAgent running for user %s", state["user_id"])
    agent = build_preference_executor(state["user_id"], _llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", "Fetch all user preference data and synthesize a PreferenceContext.")]
        })
        ctx: PreferenceContext = result["structured_response"]
        logger.info("PreferenceAgent done — origin: %s, budget: %s", ctx.origin, ctx.budget_style)
        return {
            "preference_context": ctx,
            "workflow_statuses": _status_update(
                state, "preferences", "succeeded" if _preference_has_data(ctx) else "empty"
            ),
        }
    except Exception:
        logger.error("PreferenceAgent failed", exc_info=True)
        return {
            "preference_context": PreferenceContext(),
            "workflow_statuses": _status_update(state, "preferences", "failed"),
        }
