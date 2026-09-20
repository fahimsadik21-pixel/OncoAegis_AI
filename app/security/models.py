from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.security.roles import UserRole


@dataclass
class User:
    """Application user without exposing password material in responses."""

    user_id: str
    username: str
    email: str
    password_hash: str | None = None
    role: UserRole = UserRole.VIEWER
    is_active: bool = True
    email_verified: bool = False
    provider: str = "local"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_role(self, role: UserRole) -> bool:
        return self.role == role

    def can_access_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    def to_public_dict(self) -> dict[str, object]:
        """Return a response-safe representation without password material."""

        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "provider": self.provider,
            "created_at": self.created_at.isoformat(),
        }
