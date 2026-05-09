from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai.agent import run_travel_agent
from ai.agents import PreferenceAgent
from auth.dependencies import get_current_user
from auth.models import User

router = APIRouter(prefix="/api/agent")

_preference_agent = PreferenceAgent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ChatResponse:
    preference_context = await _preference_agent.run(current_user.id)
    reply = await run_travel_agent(body.message, preference_context=preference_context)
    return ChatResponse(reply=reply)
