from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai.agent import run_travel_agent
from ai.schemas import TravelAgentStructuredResponse
from auth.dependencies import get_current_user
from auth.models import User

router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    trip_plan: TravelAgentStructuredResponse


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ChatResponse:
    trip_plan = await run_travel_agent(str(current_user.id), body.message)
    return ChatResponse(trip_plan=trip_plan)
