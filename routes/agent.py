import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agent import run_travel_agent
from ai.schemas import TravelAgentChatResponse, TransportSelection
from auth.dependencies import get_current_user
from auth.models import User
from chat.service import load_chat_history, save_chat_turn
from db import get_db_session
from trips.service import create_trip, get_trip, update_trip_from_plan


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str
    session_id: str
    transport_selection: TransportSelection | None = None
    target_trip_id: UUID | None = None


class ChatHistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _ai_message_content(row: Any) -> str:
    """Build a richer AIMessage string so the LLM can reason about prior trip plans."""
    payload = row.payload or {}
    if payload.get("response_type") == "trip_plan":
        plan = payload.get("trip_plan") or {}
        dest = plan.get("destination", "")
        days = plan.get("days", "")
        summary = plan.get("summary", "")
        budget = (plan.get("budget") or {}).get("total", "")
        parts = [row.content]
        if dest and days:
            parts.append(f"Planned: {days}-day trip to {dest}.")
        if summary:
            parts.append(summary)
        if budget:
            parts.append(f"Total budget: {budget}.")
        return " ".join(p for p in parts if p)
    return row.content


def _trip_context_message(trip: Any) -> BaseMessage:
    return SystemMessage(
        content=(
            "The user is editing an existing saved trip. Revise this trip rather than treating "
            "the request as a brand-new plan.\n\n"
            f"Trip ID: {trip.id}\n"
            f"Destination: {trip.destination}\n"
            f"Origin: {trip.origin or 'unknown'}\n"
            f"Dates: {trip.start_date or 'unknown'} to {trip.end_date or 'unknown'}\n"
            f"Days: {trip.days}\n"
            f"Travelers: {trip.travelers}\n"
            f"Summary: {trip.summary}\n"
            f"Budget: {_dump(trip.budget)}\n"
            f"Itinerary: {_dump(trip.itinerary)}\n"
            f"Selected transport: {_dump(trip.transport_options)}"
        )
    )


@router.get("/history/{session_id}", response_model=list[ChatHistoryMessage])
async def chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[ChatHistoryMessage]:
    history_rows = await load_chat_history(session, session_id, current_user.id)
    return [
        ChatHistoryMessage(
            id=str(row.id),
            role=row.role,
            content=row.content,
            created_at=row.created_at.isoformat(),
            payload=row.payload or {},
        )
        for row in history_rows
    ]


@router.post("/chat", response_model=TravelAgentChatResponse)
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> TravelAgentChatResponse:
    logger.info("Chat request [session=%s]: %s", body.session_id[:8], body.message[:80])

    history_rows = await load_chat_history(session, body.session_id, current_user.id)
    lc_history = [
        HumanMessage(content=r.content) if r.role == "user" else AIMessage(content=_ai_message_content(r))
        for r in history_rows
    ]
    target_trip = None
    if body.target_trip_id is not None:
        target_trip = await get_trip(session, current_user.id, body.target_trip_id)
        if target_trip is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
        lc_history.insert(0, _trip_context_message(target_trip))

    response = await run_travel_agent(
        current_user.id,
        body.message,
        body.session_id,
        history=lc_history,
        transport_selection=body.transport_selection,
    )

    if body.target_trip_id is not None and response.response_type == "trip_plan" and response.trip_plan is not None:
        response.trip_plan = response.trip_plan.model_copy(update={"id": str(body.target_trip_id)})

    user_payload = {
        "target_trip_id": str(body.target_trip_id) if body.target_trip_id else None,
        "transport_selection": _dump(body.transport_selection) if body.transport_selection else None,
    }
    assistant_payload = response.model_dump(mode="json", by_alias=True)

    await save_chat_turn(
        session,
        body.session_id,
        current_user.id,
        body.message,
        response.assistant_message,
        user_payload={k: v for k, v in user_payload.items() if v is not None},
        assistant_payload=assistant_payload,
    )

    if response.response_type == "trip_plan" and response.trip_plan is not None:
        if body.target_trip_id is not None:
            updated_trip = await update_trip_from_plan(session, current_user.id, body.target_trip_id, response.trip_plan)
            if updated_trip is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
        else:
            await create_trip(session, current_user.id, response.trip_plan)

    logger.info("Chat response ready [session=%s]", body.session_id[:8])
    return response
