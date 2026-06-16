import logging

from fastapi import APIRouter, Depends
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agent import run_travel_agent
from ai.schemas import TravelAgentChatResponse
from auth.dependencies import get_current_user
from auth.models import User
from chat.service import load_chat_history, save_chat_turn
from db import get_db_session
from trips.service import create_trip

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str
    session_id: str


@router.post("/chat", response_model=TravelAgentChatResponse)
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> TravelAgentChatResponse:
    logger.info("Chat request [session=%s]: %s", body.session_id[:8], body.message[:80])

    history_rows = await load_chat_history(session, body.session_id)
    lc_history = [
        HumanMessage(content=r.content) if r.role == "user" else AIMessage(content=r.content)
        for r in history_rows
    ]

    response = await run_travel_agent(current_user.id, body.message, history=lc_history)

    await save_chat_turn(session, body.session_id, current_user.id, body.message, response.assistant_message)

    if response.response_type == "trip_plan" and response.trip_plan is not None:
        await create_trip(session, current_user.id, response.trip_plan)

    logger.info("Chat response ready [session=%s]", body.session_id[:8])
    return response
