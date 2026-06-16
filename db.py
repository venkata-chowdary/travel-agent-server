from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import settings


class Base(DeclarativeBase):
    pass


def _normalize_database_url(database_url: str) -> str:
    normalized_url = database_url.strip()

    if normalized_url.startswith("postgres://"):
        normalized_url = normalized_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif normalized_url.startswith("postgresql://"):
        normalized_url = normalized_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    parsed_url = urlparse(normalized_url)
    query_params = dict(parse_qsl(parsed_url.query, keep_blank_values=True))

    sslmode = query_params.pop("sslmode", None)
    if sslmode and "ssl" not in query_params:
        query_params["ssl"] = sslmode

    query_params.pop("channel_binding", None)

    return urlunparse(parsed_url._replace(query=urlencode(query_params)))


DATABASE_URL = _normalize_database_url(settings.database_url)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    import auth.models  # noqa: F401
    import trips.models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await engine.dispose()
