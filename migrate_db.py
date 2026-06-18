import asyncio

from sqlalchemy import text

from db import close_db, engine, ensure_schema_compatibility


async def main():
    try:
        async with engine.begin() as conn:
            print("Running migration to add preferences and has_seen_preferences_dialog to users table...")
            await conn.execute(
                text("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb;")
            )
            await conn.execute(text("""
                UPDATE users
                SET preferences = jsonb_set(preferences, '{origin}', preferences->'home_city', true)
                WHERE preferences ? 'home_city'
                  AND NOT (preferences ? 'origin')
                  AND preferences->'home_city' IS NOT NULL
                  AND preferences->'home_city' <> 'null'::jsonb;
            """))
            await conn.execute(
                text("ALTER TABLE users ADD COLUMN IF NOT EXISTS has_seen_preferences_dialog BOOLEAN DEFAULT FALSE;")
            )
            print("Running migration to add trips table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trips (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    destination VARCHAR(255) NOT NULL,
                    origin VARCHAR(255),
                    start_date VARCHAR(64),
                    end_date VARCHAR(64),
                    days INTEGER NOT NULL,
                    travelers INTEGER NOT NULL DEFAULT 1,
                    status VARCHAR(32) NOT NULL DEFAULT 'planning',
                    cover_emoji VARCHAR(32) NOT NULL DEFAULT 'plane',
                    summary TEXT NOT NULL,
                    budget JSONB NOT NULL DEFAULT '{}'::jsonb,
                    itinerary JSONB NOT NULL DEFAULT '[]'::jsonb,
                    hotel_options JSONB NOT NULL DEFAULT '[]'::jsonb,
                    flight_options JSONB NOT NULL DEFAULT '[]'::jsonb,
                    transport_options JSONB NOT NULL DEFAULT '[]'::jsonb,
                    daily_forecast JSONB NOT NULL DEFAULT '[]'::jsonb,
                    trip_risks JSONB NOT NULL DEFAULT '[]'::jsonb,
                    verification_tips JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_user_id ON trips(user_id);"))
            print("Running compatibility migration for existing tables...")
            await ensure_schema_compatibility(conn)
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_status ON trips(status);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_created_at ON trips(created_at);"))
            print("Running migration to add chat_messages table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(64) NOT NULL,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role VARCHAR(16) NOT NULL,
                    content TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """))
            await conn.execute(
                text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb;")
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id ON chat_messages(session_id);")
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_chat_messages_user_id ON chat_messages(user_id);")
            )
            print("Migration complete!")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
