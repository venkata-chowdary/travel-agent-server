import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.middleware import AuthContextMiddleware
from auth.routes import router as auth_router
from config import settings
from db import close_db, init_db
from routes.agent import router as agent_router


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log("[server] Starting up...")
    await init_db()
    log("[server] Database ready — http://127.0.0.1:8000")
    yield
    log("[server] Shutting down")
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
app.include_router(agent_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    print("hello")
    log("[server] GET / hit")
    return {
        "status": "ok",
        "service": "autonomous-travel-companion-api",
    }
