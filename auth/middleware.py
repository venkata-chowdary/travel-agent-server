import sys
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from auth.security import extract_bearer_token, decode_access_token
from db import SessionLocal
from auth.models import User
from sqlalchemy import select

class AuthContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"[middleware] {request.method} {request.url.path}", file=sys.stderr, flush=True)
        token = extract_bearer_token(request.headers.get("Authorization"))
        if token:
            try:
                payload = decode_access_token(token)
                user_id = payload.get("user_id")
                async with SessionLocal() as session:
                    result = await session.execute(select(User).where(User.id == user_id))
                    user = result.scalars().first()
                    request.state.user = user
                    request.state.auth_error = None
            except Exception as e:  # covers PyJWTError, DB errors, malformed tokens
                print(f"[auth] Token validation failed: {e}", file=sys.stderr, flush=True)
                request.state.user = None
                request.state.auth_error = {"status_code": 401, "detail": str(e)}
        else:
            request.state.user = None
            request.state.auth_error = None

        response = await call_next(request)
        return response
