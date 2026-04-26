import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from config import settings


ACCESS_TOKEN_EXPIRE_SECONDS = settings.jwt_expire_days * 24 * 60 * 60


class TokenValidationError(Exception):
    pass


class TokenExpiredError(TokenValidationError):
    pass


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=settings.bcrypt_rounds)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except ValueError:
        return False


def create_access_token(*, user_id: uuid.UUID) -> str:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(days=settings.jwt_expire_days)
    payload = {
        "sub": str(user_id),
        "user_id": str(user_id),
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }

    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except ExpiredSignatureError as exc:
        raise TokenExpiredError("Access token has expired.") from exc
    except InvalidTokenError as exc:
        raise TokenValidationError("Invalid access token.") from exc

    user_id = payload.get("user_id")
    if not user_id:
        raise TokenValidationError("Access token is missing the user_id claim.")

    try:
        uuid.UUID(str(user_id))
    except ValueError as exc:
        raise TokenValidationError("Access token contains an invalid user_id.") from exc

    return payload


def extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None

    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise TokenValidationError("Authorization header must use the Bearer scheme.")

    return token.strip()
