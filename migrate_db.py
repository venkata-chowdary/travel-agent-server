import asyncio
from db import engine
from sqlalchemy import text

async def main():
    async with engine.begin() as conn:
        print("Running migration to add preferences and has_seen_preferences_dialog to users table...")
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb;"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS has_seen_preferences_dialog BOOLEAN DEFAULT FALSE;"))
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
                daily_forecast JSONB NOT NULL DEFAULT '[]'::jsonb,
                trip_risks JSONB NOT NULL DEFAULT '[]'::jsonb,
                verification_tips JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_user_id ON trips(user_id);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_status ON trips(status);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_created_at ON trips(created_at);"))
        print("Migration complete!")

if __name__ == "__main__":
    asyncio.run(main())
