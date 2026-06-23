from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
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


SCHEMA_COMPATIBILITY_STATEMENTS = (
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb",
    (
        "UPDATE users SET preferences = jsonb_set(preferences, '{origin}', preferences->'home_city', true) "
        "WHERE preferences ? 'home_city' AND NOT (preferences ? 'origin') "
        "AND preferences->'home_city' IS NOT NULL AND preferences->'home_city' <> 'null'::jsonb"
    ),
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS has_seen_preferences_dialog BOOLEAN DEFAULT FALSE",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS origin VARCHAR(255)",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS start_date VARCHAR(64)",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS end_date VARCHAR(64)",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS travelers INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'planning'",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS cover_emoji VARCHAR(32) NOT NULL DEFAULT 'plane'",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS hotel_options JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS hotel_status VARCHAR(32) NOT NULL DEFAULT 'not_searched'",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS flight_options JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS transport_options JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS daily_forecast JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS trip_risks JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS verification_tips JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created ON chat_messages (session_id, created_at)",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS transport_status VARCHAR(32) NOT NULL DEFAULT 'not_searched'",
    "ALTER TABLE trips ADD COLUMN IF NOT EXISTS session_id VARCHAR(255)",
    "CREATE INDEX IF NOT EXISTS ix_trips_session_id ON trips (session_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_trips_draft_user_session ON trips (user_id, session_id) WHERE status = 'draft'",
    """
    CREATE TABLE IF NOT EXISTS trip_hotel_options (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id VARCHAR(255) NOT NULL,
        trip_id UUID REFERENCES trips(id) ON DELETE CASCADE,
        option_id VARCHAR(255) NOT NULL,
        destination VARCHAR(255) NOT NULL,
        checkin VARCHAR(64) NOT NULL,
        checkout VARCHAR(64) NOT NULL,
        nights INTEGER NOT NULL,
        travelers INTEGER NOT NULL DEFAULT 1,
        name VARCHAR(255) NOT NULL,
        provider VARCHAR(255) NOT NULL,
        area VARCHAR(255),
        hotel_type VARCHAR(64) NOT NULL,
        price_per_night INTEGER NOT NULL,
        total_price INTEGER NOT NULL,
        rating DOUBLE PRECISION NOT NULL,
        amenities JSONB NOT NULL DEFAULT '[]'::jsonb,
        distance_from_center_km DOUBLE PRECISION,
        available_rooms INTEGER NOT NULL DEFAULT 0,
        refundable BOOLEAN NOT NULL DEFAULT FALSE,
        breakfast_included BOOLEAN NOT NULL DEFAULT FALSE,
        status VARCHAR(20) NOT NULL DEFAULT 'available',
        is_recommended BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_trip_hotel_session_search_option
            UNIQUE (session_id, destination, checkin, checkout, option_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_trip_hotel_options_session_id ON trip_hotel_options (session_id)",
    "CREATE INDEX IF NOT EXISTS ix_trip_hotel_options_trip_id ON trip_hotel_options (trip_id)",
    """
    INSERT INTO trip_hotel_options (
        session_id, trip_id, option_id, destination, checkin, checkout, nights, travelers,
        name, provider, area, hotel_type, price_per_night, total_price, rating, amenities,
        distance_from_center_km, available_rooms, refundable, breakfast_included,
        status, is_recommended, created_at
    )
    SELECT
        COALESCE(t.session_id, t.id::text),
        t.id,
        hotel->>'id',
        t.destination,
        COALESCE(t.start_date, ''),
        COALESCE(t.end_date, ''),
        t.days,
        t.travelers,
        hotel->>'name',
        COALESCE(hotel->>'provider', 'MockStay'),
        hotel->>'area',
        COALESCE(hotel->>'hotel_type', 'hotel'),
        COALESCE(NULLIF(hotel->>'price_per_night', ''), '0')::integer,
        COALESCE(NULLIF(hotel->>'total_price', ''), '0')::integer,
        COALESCE(NULLIF(hotel->>'rating', ''), '0')::double precision,
        COALESCE(hotel->'amenities', '[]'::jsonb),
        NULLIF(hotel->>'distance_from_center_km', '')::double precision,
        COALESCE(NULLIF(hotel->>'available_rooms', ''), '0')::integer,
        COALESCE((hotel->>'refundable')::boolean, FALSE),
        COALESCE((hotel->>'breakfast_included')::boolean, FALSE),
        'selected',
        TRUE,
        t.created_at
    FROM trips t
    CROSS JOIN LATERAL jsonb_array_elements(t.hotel_options) AS hotel
    WHERE jsonb_typeof(t.hotel_options) = 'array'
      AND jsonb_array_length(t.hotel_options) > 0
      AND hotel ? 'id'
      AND hotel ? 'name'
    ON CONFLICT ON CONSTRAINT uq_trip_hotel_session_search_option DO NOTHING
    """,
    """
    UPDATE trips
    SET hotel_status = 'selected'
    WHERE hotel_status = 'not_searched'
      AND jsonb_typeof(hotel_options) = 'array'
      AND jsonb_array_length(hotel_options) > 0
    """,
)


async def ensure_schema_compatibility(connection: AsyncConnection) -> None:
    for statement in SCHEMA_COMPATIBILITY_STATEMENTS:
        await connection.execute(text(statement))


async def init_db() -> None:
    import auth.models  # noqa: F401
    import chat.models  # noqa: F401
    import trips.models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await ensure_schema_compatibility(connection)


async def close_db() -> None:
    await engine.dispose()
