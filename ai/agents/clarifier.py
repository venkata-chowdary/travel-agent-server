from __future__ import annotations

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from ai.helpers import get_llm
from ai.prompts import CLARIFIER_PROMPT
from ai.schemas import TravelAgentChatResponse
from ai.state import ClarificationDecision, TravelState, build_state_summary, set_status
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


async def clarifier_node(state: TravelState) -> dict:
    logger.info("Clarifier running — LLM deciding whether clarification is needed")
    try:
        decision: ClarificationDecision = await _llm.with_structured_output(
            ClarificationDecision, method="json_schema"
        ).ainvoke([
            SystemMessage(content=CLARIFIER_PROMPT),
            *(state.get("messages") or []),
            HumanMessage(content=state["user_message"]),
            *build_state_summary(state),
        ])
    except OutputParserException as exc:
        logger.warning("Clarifier output parse failed; assuming clarification needed: %s", exc)
        decision = ClarificationDecision(
            needs_clarification=True,
            questions=["Could you clarify your trip details — destination, origin, and how many days?"],
            assistant_message="I had trouble understanding the request. Could you clarify your trip details — where you're going, where you're leaving from, and how many days you'd like?",
        )

    if decision.needs_clarification:
        logger.info("Clarifier asking %s question(s)", len(decision.questions))
        return {
            "clarification_checked": True,
            "workflow_statuses": set_status(state, "clarification", "waiting_for_user"),
            "clarification_response": TravelAgentChatResponse(
                response_type="clarification",
                assistant_message=decision.assistant_message,
                questions=decision.questions,
            ),
        }

    logger.info("Clarifier passed; enough detail to plan")
    return {
        "clarification_checked": True,
        "workflow_statuses": set_status(state, "clarification", "succeeded"),
    }
