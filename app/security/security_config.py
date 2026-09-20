from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_local_env() -> None:
    """Load simple local `.env` values without overriding real environment vars."""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#") or "=" not in candidate:
            continue
        key, value = candidate.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_local_env()

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "oncoaegis-ai"
DEFAULT_JWT_SECRET = "oncoaegis-local-development-secret-change-before-deployment"

ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

MAX_LOGIN_ATTEMPTS = 5
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
PBKDF2_ITERATIONS = 310_000

AUTH_DATABASE_PATH = Path(
    os.getenv(
        "ONCOAEGIS_AUTH_DATABASE_PATH",
        str(PROJECT_ROOT / "outputs" / "auth.sqlite3"),
    )
)

SECURITY_VERSION = "2.0"


def get_jwt_secret() -> str:
    """Return the configured signing secret, with a local-only fallback."""

    return os.getenv("ONCOAEGIS_JWT_SECRET_KEY", DEFAULT_JWT_SECRET)


def jwt_secret_is_configured() -> bool:
    return bool(os.getenv("ONCOAEGIS_JWT_SECRET_KEY"))
