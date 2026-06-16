import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chat.models import ChatMessage

HISTORY_LIMIT = 20


async def load_chat_history(session: AsyncSession, session_id: str) -> list[ChatMessage]:
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(HISTORY_LIMIT)
    )
    return list(result.scalars().all())


async def save_chat_turn(
    session: AsyncSession,
    session_id: str,
    user_id: uuid.UUID,
    user_content: str,
    assistant_content: str,
) -> None:
    session.add_all([
        ChatMessage(session_id=session_id, user_id=user_id, role="user", content=user_content),
        ChatMessage(session_id=session_id, user_id=user_id, role="assistant", content=assistant_content),
    ])
    await session.commit()
