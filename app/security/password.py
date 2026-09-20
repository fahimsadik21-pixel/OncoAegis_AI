from __future__ import annotations

import hashlib
import hmac
import os

from app.security.security_config import (
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    PBKDF2_ITERATIONS,
)


_FORMAT = "pbkdf2_sha256"
_LEGACY_ITERATIONS = 120_000


def validate_password(password: str) -> str:
    if not isinstance(password, str):
        raise ValueError("Password must be text")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
        )
    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValueError(
            f"Password must be at most {PASSWORD_MAX_LENGTH} characters"
        )
    return password


def hash_password(password: str) -> str:
    """Create a salted PBKDF2 password hash suitable for database storage."""

    validate_password(password)
    salt = os.urandom(16)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{_FORMAT}${PBKDF2_ITERATIONS}${salt.hex()}${derived_key.hex()}"


def verify_password(password: str, stored_hash: str | None) -> bool:
    """Verify both current and pre-auth-phase password hash formats."""

    if not isinstance(password, str) or not stored_hash:
        return False

    try:
        parts = stored_hash.split("$")
        if len(parts) == 4 and parts[0] == _FORMAT:
            _, iterations_text, salt_hex, hash_hex = parts
            iterations = int(iterations_text)
        elif len(parts) == 2:
            salt_hex, hash_hex = parts
            iterations = _LEGACY_ITERATIONS
        else:
            return False

        if iterations < 100_000 or iterations > 10_000_000:
            return False
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        new_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(new_hash, expected_hash)
    except (TypeError, ValueError):
        return False
