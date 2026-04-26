from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.middleware import AuthContextMiddleware
from auth.routes import router as auth_router
from config import settings
from db import close_db, init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="Autonomous Travel Companion API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthContextMiddleware)

app.include_router(auth_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "autonomous-travel-companion-api",
    }
