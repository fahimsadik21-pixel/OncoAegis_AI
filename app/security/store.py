"""SQLite persistence for users, OAuth identities and revocable sessions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import re
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Iterator

from app.security.models import User
from app.security.roles import UserRole
from app.security.security_config import AUTH_DATABASE_PATH


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
OAUTH_PROVIDERS = {"google", "apple"}


class AuthStoreError(RuntimeError):
    """Base error for authentication persistence failures."""


class DuplicateUserError(AuthStoreError):
    """Email or username is already registered."""


class UserNotFoundError(AuthStoreError):
    """Requested user does not exist."""


def normalize_email(email: object) -> str:
    value = str(email or "").strip().casefold()
    if len(value) > 254 or not EMAIL_PATTERN.fullmatch(value):
        raise ValueError("A valid email address is required")
    return value


def normalize_username(username: object | None) -> str | None:
    if username is None or not str(username).strip():
        return None
    value = str(username).strip()
    if not USERNAME_PATTERN.fullmatch(value):
        raise ValueError(
            "Username must be 3-32 characters using letters, numbers, '.', '_' or '-'"
        )
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class AuthStore:
    """Small, thread-safe SQLite store for authentication state."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path if db_path is not None else AUTH_DATABASE_PATH
        self.db_path = Path(configured) if str(configured) != ":memory:" else None
        self._memory_connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        if self.db_path is None:
            self._memory_connection = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._memory_connection.row_factory = sqlite3.Row
            self._initialize(self._memory_connection)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                self._initialize(connection)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            try:
                yield self._memory_connection
                self._memory_connection.commit()
            except Exception:
                self._memory_connection.rollback()
                raise
            return

        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS auth_users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                email_verified INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT 'local',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_oauth_accounts (
                provider TEXT NOT NULL,
                provider_subject TEXT NOT NULL,
                user_id TEXT NOT NULL REFERENCES auth_users(user_id),
                created_at TEXT NOT NULL,
                PRIMARY KEY (provider, provider_subject)
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                jti TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES auth_users(user_id),
                token_type TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                ON auth_sessions(user_id);

            CREATE TABLE IF NOT EXISTS auth_oauth_states (
                state TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                used_at TEXT
            );
            """
        )
        connection.commit()

    @staticmethod
    def _row_to_user(row: sqlite3.Row | None) -> User | None:
        if row is None:
            return None
        return User(
            user_id=str(row["user_id"]),
            username=str(row["username"]),
            email=str(row["email"]),
            password_hash=row["password_hash"],
            role=UserRole(str(row["role"])),
            is_active=bool(row["is_active"]),
            email_verified=bool(row["email_verified"]),
            provider=str(row["provider"]),
            created_at=_parse_datetime(str(row["created_at"])),
            updated_at=_parse_datetime(str(row["updated_at"])),
        )

    def _unique_username(self, connection: sqlite3.Connection, requested: str | None, email: str) -> str:
        base = requested or re.sub(r"[^A-Za-z0-9_.-]", "", email.split("@", 1)[0])
        base = (base or "user")[:32]
        if len(base) < 3:
            base = f"user-{base}"[:32]
        candidate = base
        counter = 2
        while connection.execute(
            "SELECT 1 FROM auth_users WHERE username = ? COLLATE NOCASE",
            (candidate,),
        ).fetchone():
            suffix = f"-{counter}"
            candidate = f"{base[:32 - len(suffix)]}{suffix}"
            counter += 1
        return candidate

    def create_user(
        self,
        *,
        email: str,
        password_hash: str | None,
        username: str | None = None,
        role: UserRole = UserRole.VIEWER,
        provider: str = "local",
        email_verified: bool = False,
    ) -> User:
        normalized_email = normalize_email(email)
        normalized_username = normalize_username(username)
        role = UserRole(role)
        if provider not in {"local", *OAUTH_PROVIDERS}:
            raise ValueError("Unsupported authentication provider")
        now = _utc_now()
        user_id = f"usr_{uuid.uuid4().hex}"
        with self._lock, self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM auth_users WHERE email = ? COLLATE NOCASE",
                (normalized_email,),
            ).fetchone():
                raise DuplicateUserError("Email is already registered")
            final_username = self._unique_username(
                connection, normalized_username, normalized_email
            )
            if normalized_username and final_username != normalized_username:
                raise DuplicateUserError("Username is already registered")
            try:
                connection.execute(
                    """
                    INSERT INTO auth_users(
                        user_id, username, email, password_hash, role,
                        is_active, email_verified, provider, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        final_username,
                        normalized_email,
                        password_hash,
                        role.value,
                        int(email_verified),
                        provider,
                        _iso(now),
                        _iso(now),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateUserError("Email or username is already registered") from exc
            return User(
                user_id=user_id,
                username=final_username,
                email=normalized_email,
                password_hash=password_hash,
                role=role,
                email_verified=email_verified,
                provider=provider,
                created_at=now,
                updated_at=now,
            )

    def get_user_by_email(self, email: str) -> User | None:
        normalized_email = normalize_email(email)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_users WHERE email = ? COLLATE NOCASE",
                (normalized_email,),
            ).fetchone()
        return self._row_to_user(row)

    def get_user_by_id(self, user_id: str) -> User | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id = ?", (str(user_id),)
            ).fetchone()
        return self._row_to_user(row)

    def get_user_by_oauth(self, *, provider: str, provider_subject: str) -> User | None:
        if provider not in OAUTH_PROVIDERS or not provider_subject.strip():
            raise ValueError("Invalid OAuth identity")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT users.* FROM auth_users AS users
                INNER JOIN auth_oauth_accounts AS accounts
                    ON accounts.user_id = users.user_id
                WHERE accounts.provider = ? AND accounts.provider_subject = ?
                """,
                (provider, provider_subject.strip()),
            ).fetchone()
        return self._row_to_user(row)

    def link_oauth_identity(
        self, *, provider: str, provider_subject: str, user_id: str
    ) -> None:
        if provider not in OAUTH_PROVIDERS or not provider_subject.strip():
            raise ValueError("Invalid OAuth identity")
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO auth_oauth_accounts(
                        provider, provider_subject, user_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (provider, provider_subject.strip(), user_id, _iso(_utc_now())),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateUserError(
                    "OAuth identity is already linked to an account"
                ) from exc

    def create_session(
        self,
        *,
        jti: str,
        user_id: str,
        token_type: str,
        expires_at: datetime,
    ) -> None:
        if token_type not in {"access", "refresh"}:
            raise ValueError("Unsupported session token type")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO auth_sessions(
                    jti, user_id, token_type, expires_at, created_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (jti, user_id, token_type, _iso(expires_at), _iso(_utc_now())),
            )

    def is_session_active(self, *, jti: str, user_id: str, token_type: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT expires_at, revoked_at FROM auth_sessions
                WHERE jti = ? AND user_id = ? AND token_type = ?
                """,
                (jti, user_id, token_type),
            ).fetchone()
        return bool(
            row
            and row["revoked_at"] is None
            and _parse_datetime(str(row["expires_at"])) > _utc_now()
        )

    def revoke_session(self, jti: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE jti = ?",
                (_iso(_utc_now()), jti),
            )

    def revoke_all_sessions(self, user_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE auth_sessions SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (_iso(_utc_now()), user_id),
            )

    def create_oauth_state(
        self, provider: str, *, expires_in_seconds: int = 600
    ) -> str:
        if provider not in OAUTH_PROVIDERS:
            raise ValueError("Unsupported OAuth provider")
        state = uuid.uuid4().hex + uuid.uuid4().hex
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_oauth_states(state, provider, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    state,
                    provider,
                    _iso(now + timedelta(seconds=expires_in_seconds)),
                    _iso(now),
                ),
            )
        return state

    def oauth_state_is_valid(self, *, state: str, provider: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT expires_at, used_at FROM auth_oauth_states
                WHERE state = ? AND provider = ?
                """,
                (state, provider),
            ).fetchone()
        return bool(
            row
            and row["used_at"] is None
            and _parse_datetime(str(row["expires_at"])) > _utc_now()
        )

    def consume_oauth_state(self, *, state: str, provider: str) -> bool:
        if not self.oauth_state_is_valid(state=state, provider=provider):
            return False
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE auth_oauth_states SET used_at = ?
                WHERE state = ? AND provider = ? AND used_at IS NULL
                """,
                (_iso(_utc_now()), state, provider),
            )
        return True


_default_auth_store: AuthStore | None = None
_default_auth_store_lock = threading.Lock()


def get_auth_store() -> AuthStore:
    global _default_auth_store
    if _default_auth_store is None:
        with _default_auth_store_lock:
            if _default_auth_store is None:
                _default_auth_store = AuthStore()
    return _default_auth_store


def set_auth_store(store: AuthStore | None) -> None:
    """Replace the process store for isolated tests or deployments."""

    global _default_auth_store
    with _default_auth_store_lock:
        _default_auth_store = store
