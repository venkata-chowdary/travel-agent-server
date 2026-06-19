from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from ai.helpers import get_llm
from ai.prompts import PREFERENCE_AGENT_SYSTEM_PROMPT
from ai.schemas.preferences import PreferenceContext
from ai.schemas.signal import AgentSignal
from ai.state import TravelState, _origin_from_state, _preference_has_data, _status_update
from ai.tools.preference_tools import make_preference_tools
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


def build_preference_executor(user_id: str, llm: BaseChatModel):
    return create_react_agent(
        llm,
        make_preference_tools(user_id),
        prompt=PREFERENCE_AGENT_SYSTEM_PROMPT,
        response_format=PreferenceContext,
    )


def _preference_signal(ctx: PreferenceContext, state: TravelState) -> AgentSignal:
    if ctx.memory_confidence < 0.3:
        return AgentSignal(
            signal_type="data_sparse",
            severity="medium",
            message=(
                f"User has very sparse preference data (confidence: {ctx.memory_confidence:.2f}). "
                "Budget, travel style, and origin are largely unknown. "
                "Targeted clarification would significantly improve the plan."
            ),
        )
    if not ctx.origin and not _origin_from_state(state):
        return AgentSignal(
            signal_type="clarification_needed",
            severity="medium",
            message=(
                "I couldn't determine the user's home city from their profile or trip history. "
                "Origin is needed for transport search and budget estimates."
            ),
        )
    return AgentSignal(
        signal_type="no_action_needed",
        severity="low",
        message=(
            f"Preference profile collected (confidence: {ctx.memory_confidence:.2f}). "
            f"Travel style: {ctx.travel_style or 'unknown'}, budget: {ctx.budget_style or 'unknown'}."
        ),
    )


async def preference_agent_node(state: TravelState) -> dict:
    logger.info("PreferenceAgent running for user %s", state["user_id"])
    agent = build_preference_executor(state["user_id"], _llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", "Fetch all user preference data and synthesize a PreferenceContext.")]
        })
        ctx: PreferenceContext = result["structured_response"]
        signal = _preference_signal(ctx, state)
        logger.info(
            "PreferenceAgent done — origin: %s, budget: %s | signal: %s",
            ctx.origin, ctx.budget_style, signal.signal_type,
        )
        return {
            "preference_context": ctx,
            "preference_signal": signal,
            "workflow_statuses": _status_update(
                state, "preferences", "succeeded" if _preference_has_data(ctx) else "empty"
            ),
        }
    except Exception:
        logger.error("PreferenceAgent failed", exc_info=True)
        return {
            "preference_context": PreferenceContext(),
            "preference_signal": AgentSignal(
                signal_type="data_sparse",
                severity="medium",
                message="Preference agent failed to fetch user data. Proceeding with defaults.",
            ),
            "workflow_statuses": _status_update(state, "preferences", "failed"),
        }
