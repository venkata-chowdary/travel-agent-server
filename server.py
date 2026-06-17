import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.agent import close_agent_checkpointing, init_agent_checkpointing
from auth.middleware import AuthContextMiddleware
from auth.routes import router as auth_router
from config import settings
from db import close_db, init_db
from routes.agent import router as agent_router
from trips.routes import router as trips_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting up...")
    await init_db()
    await init_agent_checkpointing()
    logger.info("Database ready — http://127.0.0.1:8000")
    yield
    logger.info("Shutting down")
    await close_agent_checkpointing()
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
app.include_router(trips_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    logger.info("GET / hit")
    return {
        "status": "ok",
        "service": "autonomous-travel-companion-api",
    }
