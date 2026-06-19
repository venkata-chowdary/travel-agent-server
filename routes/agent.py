import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ai.chat_service import AgentService
from ai.schemas import HotelSelection, TravelAgentChatResponse, TransportSelection
from auth.dependencies import get_current_user
from auth.models import User
from db import get_db_session
from chat.service import load_chat_history
from trips.schemas import TripTransportOptionResponse
from trips.service import get_session_transport_options


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str
    session_id: str
    transport_selection: TransportSelection | None = None
    hotel_selection: HotelSelection | None = None
    target_trip_id: UUID | None = None


class ChatHistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


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
    response = await AgentService(session, current_user).chat(
        message=body.message,
        session_id=body.session_id,
        transport_selection=body.transport_selection,
        hotel_selection=body.hotel_selection,
        target_trip_id=body.target_trip_id,
    )
    logger.info("Chat response ready [session=%s]", body.session_id[:8])
    return response


@router.get("/transport/{session_id}", response_model=list[TripTransportOptionResponse])
async def pending_transport_options(
    session_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[TripTransportOptionResponse]:
    """Return available (unselected) transport options for a session.

    Frontend calls this on session load to auto-resume a transport choice that was
    generated but never completed (lost connection, logout, etc.).
    """
    return await get_session_transport_options(session, session_id, current_user.id)
