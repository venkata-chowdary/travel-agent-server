import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chat.models import ChatMessage

HISTORY_LIMIT = 40


async def load_chat_history(
    session: AsyncSession,
    session_id: str,
    user_id: uuid.UUID | None = None,
) -> list[ChatMessage]:
    query = select(ChatMessage).where(ChatMessage.session_id == session_id)
    if user_id is not None:
        query = query.where(ChatMessage.user_id == user_id)

    result = await session.execute(
        query.order_by(ChatMessage.created_at.asc()).limit(HISTORY_LIMIT)
    )
    return list(result.scalars().all())


async def save_chat_turn(
    session: AsyncSession,
    session_id: str,
    user_id: uuid.UUID,
    user_content: str,
    assistant_content: str,
    user_payload: dict[str, Any] | None = None,
    assistant_payload: dict[str, Any] | None = None,
    commit: bool = True,
) -> None:
    session.add_all([
        ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=user_content,
            payload=user_payload or {},
        ),
        ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=assistant_content,
            payload=assistant_payload or {},
        ),
    ])
    if commit:
        await session.commit()
    else:
        await session.flush()
