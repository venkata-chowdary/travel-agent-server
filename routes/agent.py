from fastapi import APIRouter
from pydantic import BaseModel

from ai.agent import run_travel_agent

router = APIRouter(prefix="/api/agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    reply = await run_travel_agent(body.message)
    return ChatResponse(reply=reply)
