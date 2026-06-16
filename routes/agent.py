import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai.agent import run_travel_agent
from ai.schemas import TravelAgentStructuredResponse
from auth.dependencies import get_current_user
from auth.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    trip_plan: TravelAgentStructuredResponse


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    logger.info("Chat request: %s", body.message[:80])
    trip_plan = await run_travel_agent(current_user.id, body.message)
    logger.info("Chat response ready")
    return ChatResponse(trip_plan=trip_plan)
