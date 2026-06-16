import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from auth.dependencies import get_current_user
from auth.models import User
from auth.schemas import AuthResponse, LoginRequest, SignupRequest, UserResponse, PreferencesUpdateRequest
from auth.security import ACCESS_TOKEN_EXPIRE_SECONDS, hash_password, verify_password, create_access_token
from db import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

def build_auth_response(user: User, access_token: str) -> AuthResponse:
    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
        user=UserResponse.model_validate(user),
    )

@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AuthResponse:
    logger.info("Signup attempt for %s", payload.email)
    result = await session.execute(select(User).where(User.email == payload.email))
    existing_user = result.scalars().first()
    if existing_user:
        logger.warning("Signup failed — email already exists: %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="email already exists",
        )

    new_user = User(
        email=payload.email,
        name=payload.name,
        password_hash=hash_password(payload.password),
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    logger.info("New user created: %s (id=%s)", payload.email, new_user.id)
    access_token = create_access_token(user_id=new_user.id)
    return build_auth_response(new_user, access_token)

@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AuthResponse:
    logger.info("Login attempt for %s", payload.email)
    result = await session.execute(select(User).where(User.email == payload.email))
    user = result.scalars().first()

    if not user or not verify_password(payload.password, user.password_hash):
        logger.warning("Login failed — invalid credentials for %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("Login successful for %s (id=%s)", payload.email, user.id)
    access_token = create_access_token(user_id=user.id)
    return build_auth_response(user, access_token)

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(current_user)

@router.put("/preferences", response_model=UserResponse)
async def update_preferences(
    payload: PreferencesUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserResponse:
    if payload.preferences is not None:
        current_user.preferences = payload.preferences.model_dump()
    if payload.has_seen_preferences_dialog is not None:
        current_user.has_seen_preferences_dialog = payload.has_seen_preferences_dialog

    session.add(current_user)
    await session.commit()
    await session.refresh(current_user)
    logger.info("Preferences updated for user %s", current_user.id)
    return UserResponse.model_validate(current_user)
