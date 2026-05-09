from fastapi import APIRouter, Request
from pydantic import BaseModel

from ai.agent import run_travel_agent

router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    user = getattr(request.state, "user", None)
    user_id = str(user.id) if user else None
    reply = await run_travel_agent(user_id, body.message)
    return ChatResponse(reply=reply)
