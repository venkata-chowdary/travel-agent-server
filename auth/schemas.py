from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from ai.schemas import TravelPreferences


def _normalize_email(value: EmailStr) -> str:
    return value.strip().lower()


def _validate_password(value: str) -> str:
    if len(value.encode("utf-8")) > 72:
        raise ValueError("Password must be 72 bytes or fewer.")

    if value != value.strip():
        raise ValueError("Password cannot start or end with whitespace.")

    if not any(character.isalpha() for character in value):
        raise ValueError("Password must include at least one letter.")

    if not any(character.isdigit() for character in value):
        raise ValueError("Password must include at least one number.")

    return value


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    name: str = Field(min_length=1, max_length=255)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return _normalize_email(value)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return _validate_password(value)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Name is required.")
        return normalized


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return _normalize_email(value)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    name: str
    created_at: datetime
    preferences: TravelPreferences = Field(default_factory=TravelPreferences)
    has_seen_preferences_dialog: bool = False


class PreferencesUpdateRequest(BaseModel):
    preferences: TravelPreferences | None = None
    has_seen_preferences_dialog: bool | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
