from datetime import datetime, timedelta, timezone
import random
import re
import string
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from auth.schemas import (
    UserResponse,
    UserCreate,
    UserProfileUpdate,
    EmailVerificationRequest,
    EmailVerificationConfirm,
    PasswordChange,
)
from db import get_session
from auth.models import User
from auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user_id,
)
from email_service import send_verification_email


router = APIRouter(prefix="/auth", tags=["Auth"])
GITHUB_USERNAME_PATTERN = re.compile(
    r"^[A-Za-z\d](?:[A-Za-z\d]|-(?=[A-Za-z\d])){0,38}$"
)


def _normalize_github_username(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    if normalized.startswith("@"):
        normalized = normalized[1:]

    if "github.com" in normalized.lower():
        parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise HTTPException(
                status_code=422,
                detail="Enter a valid GitHub username or GitHub profile URL.",
            )

        path_parts = [part for part in parsed.path.split("/") if part]
        if not path_parts:
            raise HTTPException(
                status_code=422,
                detail="Enter a valid GitHub username or GitHub profile URL.",
            )
        normalized = path_parts[0]

    if not GITHUB_USERNAME_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=422,
            detail="Enter a valid GitHub username or GitHub profile URL.",
        )

    return normalized


@router.post("/register", response_model=UserResponse, status_code=201)
async def register_user(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    existing_user_query = select(User).where(User.email == user.email)
    existing_user = await session.exec(existing_user_query)

    if existing_user.first():
        raise HTTPException(400, "email already exists")

    new_user = User(email=user.email, hased_password=hash_password(user.password))

    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    await _set_and_send_verification_otp(new_user, session, background_tasks)

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        first_name=new_user.first_name,
        last_name=new_user.last_name,
        bio=new_user.bio,
        github_username=new_user.github_username,
        is_email_verified=new_user.is_email_verified,
    )


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(User).where(User.email == form_data.username)
    result = await session.exec(stmt)
    user = result.first()

    if not user or not verify_password(form_data.password, user.hased_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "bio": user.bio,
            "github_username": user.github_username,
            "is_email_verified": user.is_email_verified,
        },
    }


@router.get("/me", response_model=UserResponse)
async def get_me(
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(User).where(User.id == UUID(user_id))
    result = await session.exec(stmt)
    user = result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        bio=user.bio,
        github_username=user.github_username,
        is_email_verified=user.is_email_verified,
    )


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    payload: UserProfileUpdate,
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(User).where(User.id == UUID(user_id))
    result = await session.exec(stmt)
    user = result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.first_name is not None:
        user.first_name = payload.first_name
    if payload.last_name is not None:
        user.last_name = payload.last_name
    if payload.bio is not None:
        user.bio = payload.bio
    if payload.github_username is not None:
        user.github_username = _normalize_github_username(payload.github_username)

    session.add(user)
    await session.commit()
    await session.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        bio=user.bio,
        github_username=user.github_username,
        is_email_verified=user.is_email_verified,
    )


@router.post("/change-password")
async def change_password(
    payload: PasswordChange,
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(User).where(User.id == UUID(user_id))
    result = await session.exec(stmt)
    user = result.first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(payload.old_password, user.hased_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")

    user.hased_password = hash_password(payload.new_password)

    session.add(user)
    await session.commit()

    return {"message": "Password updated successfully"}


def _generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


async def _set_and_send_verification_otp(
    user: User,
    session: AsyncSession,
    background_tasks: BackgroundTasks,
) -> None:
    otp = _generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    user.email_verification_otp = otp
    user.email_verification_expires_at = expires_at

    session.add(user)
    await session.commit()

    background_tasks.add_task(send_verification_email, user.email, otp)


@router.post("/email/request-otp")
async def request_email_verification(
    payload: EmailVerificationRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(User).where(User.email == payload.email)
    result = await session.exec(stmt)
    user = result.first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Email already verified")

    await _set_and_send_verification_otp(user, session, background_tasks)

    return {"message": "Verification OTP sent to email"}


@router.post("/email/verify")
async def verify_email(
    payload: EmailVerificationConfirm,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(User).where(User.email == payload.email)
    result = await session.exec(stmt)
    user = result.first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_email_verified:
        return {"message": "Email already verified"}

    if (
        not user.email_verification_otp
        or not user.email_verification_expires_at
        or payload.otp != user.email_verification_otp
    ):
        raise HTTPException(status_code=400, detail="Invalid OTP")

    if datetime.now(timezone.utc) > user.email_verification_expires_at:
        raise HTTPException(status_code=400, detail="OTP has expired")

    user.is_email_verified = True
    user.email_verification_otp = None
    user.email_verification_expires_at = None

    session.add(user)
    await session.commit()

    return {"message": "Email verified successfully"}
