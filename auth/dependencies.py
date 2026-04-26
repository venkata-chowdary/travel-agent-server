from fastapi import HTTPException, Request, status

from auth.models import User


def get_current_user(request: Request) -> User:
    auth_error = getattr(request.state, "auth_error", None)
    if auth_error is not None:
        raise HTTPException(
            status_code=auth_error["status_code"],
            detail=auth_error["detail"],
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
