import asyncio
from db import engine
from sqlalchemy import text

async def main():
    async with engine.begin() as conn:
        print("Running migration to add preferences and has_seen_preferences_dialog to users table...")
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb;"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS has_seen_preferences_dialog BOOLEAN DEFAULT FALSE;"))
        print("Migration complete!")

if __name__ == "__main__":
    asyncio.run(main())
