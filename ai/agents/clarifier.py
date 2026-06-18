from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ai.helpers import GeminiClient
from ai.prompts import CLARIFIER_PROMPT
from ai.schemas import TravelAgentChatResponse
from ai.state import ClarificationDecision, TravelState, _state_summary, _status_update
from config import settings

logger = logging.getLogger(__name__)
_llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)


async def clarifier_node(state: TravelState) -> dict:
    logger.info("Clarifier running — LLM deciding whether clarification is needed")
    decision: ClarificationDecision = await _llm.with_structured_output(
        ClarificationDecision, method="json_schema"
    ).ainvoke([
        SystemMessage(content=CLARIFIER_PROMPT),
        *(state.get("messages") or []),
        HumanMessage(content=state["user_message"]),
        *_state_summary(state),
    ])

    if decision.needs_clarification:
        logger.info("Clarifier asking %s question(s)", len(decision.questions))
        return {
            "clarification_checked": True,
            "workflow_statuses": _status_update(state, "clarification", "waiting_for_user"),
            "clarification_response": TravelAgentChatResponse(
                response_type="clarification",
                assistant_message=decision.assistant_message,
                questions=decision.questions,
            ),
        }

    logger.info("Clarifier passed; enough detail to plan")
    return {
        "clarification_checked": True,
        "workflow_statuses": _status_update(state, "clarification", "succeeded"),
    }
