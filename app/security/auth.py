from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid

from jose import JWTError, jwt

from app.security.models import User
from app.security.roles import UserRole
from app.security.security_config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_ISSUER,
    REFRESH_TOKEN_EXPIRE_DAYS,
    get_jwt_secret,
)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    username: str
    role: UserRole
    email: str = ""
    is_active: bool = True


def create_user_context(
    user_id: str,
    username: str,
    role: UserRole,
    *,
    email: str = "",
    is_active: bool = True,
) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=user_id,
        username=username,
        role=role,
        email=email,
        is_active=is_active,
    )


def user_to_context(user: User) -> AuthenticatedUser:
    return create_user_context(
        user.user_id,
        user.username,
        user.role,
        email=user.email,
        is_active=user.is_active,
    )


def has_role(user: AuthenticatedUser, role: UserRole) -> bool:
    return user.role == role


def _role_value(role: UserRole | str) -> str:
    return role.value if isinstance(role, UserRole) else str(role)


def _create_token(
    user_id: str,
    role: UserRole | str,
    *,
    token_type: str,
    expires_delta: timedelta,
    username: str | None = None,
    email: str | None = None,
    jti: str | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "sub": user_id,
        "role": _role_value(role),
        "type": token_type,
        "iss": JWT_ISSUER,
        "iat": now,
        "exp": now + expires_delta,
        "jti": jti or uuid.uuid4().hex,
    }
    if username:
        payload["username"] = username
    if email:
        payload["email"] = email
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_access_token(
    user_id: str,
    role: UserRole | str,
    expires_minutes: int | None = None,
    *,
    username: str | None = None,
    email: str | None = None,
    jti: str | None = None,
) -> str:
    """Create a signed short-lived access token."""

    minutes = (
        ACCESS_TOKEN_EXPIRE_MINUTES
        if expires_minutes is None
        else int(expires_minutes)
    )
    if minutes <= 0:
        raise ValueError("Token lifetime must be positive")
    return _create_token(
        user_id,
        role,
        token_type="access",
        expires_delta=timedelta(minutes=minutes),
        username=username,
        email=email,
        jti=jti,
    )


def create_refresh_token(
    user_id: str,
    role: UserRole | str,
    expires_days: int | None = None,
    *,
    username: str | None = None,
    email: str | None = None,
    jti: str | None = None,
) -> str:
    """Create a signed refresh token used for session rotation."""

    days = REFRESH_TOKEN_EXPIRE_DAYS if expires_days is None else int(expires_days)
    if days <= 0:
        raise ValueError("Token lifetime must be positive")
    return _create_token(
        user_id,
        role,
        token_type="refresh",
        expires_delta=timedelta(days=days),
        username=username,
        email=email,
        jti=jti,
    )


def decode_token(token: str, *, expected_type: str | None = None) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
        )
        if not payload.get("sub") or not payload.get("jti"):
            return None
        if expected_type is not None and payload.get("type") != expected_type:
            return None
        return payload
    except (JWTError, TypeError, ValueError):
        return None


def decode_access_token(token: str) -> dict | None:
    return decode_token(token, expected_type="access")


def decode_refresh_token(token: str) -> dict | None:
    return decode_token(token, expected_type="refresh")


def get_user_id_from_token(token: str) -> str | None:
    payload = decode_access_token(token)
    return str(payload["sub"]) if payload else None


def get_role_from_token(token: str) -> str | None:
    payload = decode_access_token(token)
    return str(payload["role"]) if payload and payload.get("role") else None
