"""Public contracts for local and provider-backed authentication."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.security.security_config import PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(
        min_length=PASSWORD_MIN_LENGTH,
        max_length=PASSWORD_MAX_LENGTH,
    )
    username: str | None = Field(default=None, min_length=3, max_length=32)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    email_verified: bool
    provider: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ProviderStatus(BaseModel):
    provider: str
    label: str
    enabled: bool
    status: str


class OAuthStartResponse(BaseModel):
    provider: str
    authorization_url: str
    state_expires_in: int


class OAuthCallbackResponse(BaseModel):
    provider: str
    status: str
    detail: str
