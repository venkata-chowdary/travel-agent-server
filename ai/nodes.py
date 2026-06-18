from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ai.agents.preference_agent import build_preference_executor
from ai.agents.transport_agent import build_transport_choice
from ai.agents.weather_agent import _unavailable_forecast, build_weather_executor
from ai.helpers import GeminiClient, format_preferences_block, format_transport_block, format_weather_block
from ai.prompts import CLARIFIER_PROMPT, MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
from ai.schemas import (
    PreferenceContext,
    TravelAgentChatResponse,
    TravelAgentStructuredResponse,
    WeatherForecastResponse,
)
from ai.schemas.travel import TravelPlanLLMOutput
from ai.state import (
    ClarificationDecision,
    TravelState,
    _apply_transport_budget,
    _origin_from_state,
    _preference_has_data,
    _state_summary,
    _status_update,
    _transport_has_options,
    _trip_dates,
)
from config import settings

logger = logging.getLogger(__name__)
_llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)


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


async def weather_agent_node(state: TravelState) -> dict:
    destination = state["destination"]
    trip_dates = _trip_dates(state)
    logger.info("WeatherAgent running — %s, dates: %s", destination, trip_dates)
    agent = build_weather_executor(_llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", f"Get weather forecast for {destination} on these dates: {', '.join(trip_dates)}")]
        })
        forecast: WeatherForecastResponse = result["structured_response"]
        logger.info("WeatherAgent done — %s", forecast.summary[:80])
        return {
            "weather_forecast": forecast,
            "workflow_statuses": _status_update(
                state, "weather", "succeeded" if forecast.daily_forecast else "empty"
            ),
        }
    except Exception:
        logger.error("WeatherAgent failed", exc_info=True)
        return {
            "weather_forecast": _unavailable_forecast(destination),
            "workflow_statuses": _status_update(state, "weather", "failed"),
        }


async def transport_agent_node(state: TravelState) -> dict:
    origin = _origin_from_state(state)
    destination = state.get("destination")
    trip_dates = _trip_dates(state)
    if not origin or not destination:
        logger.info("TransportAgent skipped; missing origin or destination")
        return {"workflow_statuses": _status_update(state, "transport", "failed")}

    logger.info("TransportAgent running — %s to %s on %s", origin, destination, trip_dates[0])
    choice = build_transport_choice(
        origin=origin,
        destination=destination,
        start_date=trip_dates[0],
        days=state.get("trip_duration_days") or len(trip_dates),
        travelers=1,
        preferences=state.get("preference_context"),
    )
    logger.info(
        "TransportAgent found %s outbound and %s return option(s)",
        len(choice.outbound_options), len(choice.return_options),
    )
    return {
        "transport_choice": choice,
        "workflow_statuses": _status_update(
            state, "transport", "succeeded" if _transport_has_options(choice) else "empty"
        ),
    }


async def planner_node(state: TravelState) -> dict:
    logger.info("Planner generating trip plan...")
    date_line = f"\nToday is {datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')} (UTC)."
    system_prompt = (
        MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
        + date_line
        + format_preferences_block(state.get("preference_context"))
        + format_weather_block(state.get("weather_forecast"))
        + format_transport_block(state.get("selected_transport_options"))
    )
    messages = [SystemMessage(content=system_prompt)]
    if state.get("messages"):
        messages.extend(state["messages"])
    messages.append(HumanMessage(content=state["user_message"]))

    plan: TravelPlanLLMOutput = await _llm.with_structured_output(
        TravelPlanLLMOutput, method="json_schema"
    ).ainvoke(messages)

    try:
        result = TravelAgentStructuredResponse.model_validate(plan.model_dump())
    except ValidationError as exc:
        logger.error("Planner schema conversion failed: %s\nRaw plan: %s", exc, plan.model_dump())
        raise RuntimeError("The planner produced an invalid response. Please try again.") from exc

    updates: dict = {}
    origin = _origin_from_state(state)
    if origin:
        updates["origin"] = origin

    trip_dates = _trip_dates(state)
    if trip_dates:
        updates["start_date"] = trip_dates[0]
        updates["end_date"] = trip_dates[-1]

    wf = state.get("weather_forecast")
    if wf and wf.daily_forecast:
        updates["daily_forecast"] = wf.daily_forecast
        updates["trip_risks"] = wf.trip_risks
        updates["requires_replanning"] = wf.requires_replanning
        updates["weather_summary"] = wf.summary

    if updates:
        result = result.model_copy(update=updates)

    result = _apply_transport_budget(result, state.get("selected_transport_options"))
    logger.info("Planner done — %s, %s day(s), budget %s", result.destination, result.days, result.budget.total)
    return {"structured_response": result}
