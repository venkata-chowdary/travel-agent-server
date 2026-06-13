import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


def _require_env(name: str, *fallback_names: str) -> str:
    for candidate in (name, *fallback_names):
        value = os.getenv(candidate)
        if value and value.strip():
            return value.strip()

    if fallback_names:
        fallbacks = ", ".join(fallback_names)
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"You can also provide one of: {fallbacks}."
        )

    raise RuntimeError(f"Missing required environment variable {name}.")


def _get_non_negative_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed_value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc
    if parsed_value < 0:
        raise RuntimeError(f"{name} must be >= 0.")
    return parsed_value


def _get_positive_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()

    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc

    if parsed_value < minimum:
        raise RuntimeError(f"{name} must be greater than or equal to {minimum}.")

    return parsed_value


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_jwt_secret(secret: str) -> str:
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters long.")

    return secret


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_expire_days: int
    bcrypt_rounds: int
    cors_origins: list[str]
    llm_model: str
    llm_temperature: float


def load_settings() -> Settings:
    return Settings(
        database_url=_require_env("DATABASE_URL"),
        jwt_secret_key=_validate_jwt_secret(_require_env("JWT_SECRET_KEY", "SECRET_KEY")),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256").strip() or "HS256",
        jwt_expire_days=_get_positive_int("JWT_EXPIRE_DAYS", default=7, minimum=1),
        bcrypt_rounds=_get_positive_int("BCRYPT_ROUNDS", default=12, minimum=12),
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS", "http://localhost:3000")),
        llm_model="gemini-3-flash-preview",
        llm_temperature=_get_non_negative_float("LLM_TEMPERATURE", default=0.0),
    )


settings = load_settings()
