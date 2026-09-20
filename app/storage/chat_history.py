"""Privacy-aware, user-scoped conversation history for the product UI.

Only the conversation transcript and structured assistant metadata are stored.
Uploaded medical files are never persisted by this store.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Any, Iterator

from app.security.security_config import AUTH_DATABASE_PATH


MAX_TITLE_LENGTH = 120
MAX_MESSAGE_LENGTH = 30_000
MAX_METADATA_LENGTH = 80_000


class ChatHistoryError(RuntimeError):
    """Base error for chat-history persistence failures."""


class ChatConversationNotFound(ChatHistoryError):
    """A conversation is missing or belongs to another user."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistoryStore:
    """Thread-safe SQLite storage with an explicit user ownership boundary."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path if db_path is not None else AUTH_DATABASE_PATH
        self.db_path = Path(configured) if str(configured) != ":memory:" else None
        self._memory_connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        if self.db_path is None:
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS chat_conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_updated
                ON chat_conversations(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                kind TEXT NOT NULL CHECK(kind IN ('text', 'analysis')),
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
                ON chat_messages(conversation_id, created_at ASC);
            """
        )
        connection.commit()

    @staticmethod
    def _safe_text(value: object, *, limit: int, label: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ChatHistoryError(f"{label} cannot be empty")
        if len(text) > limit:
            raise ChatHistoryError(f"{label} is too long")
        return text

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any] | None) -> str:
        payload = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        if len(payload) > MAX_METADATA_LENGTH:
            raise ChatHistoryError("Conversation metadata is too large")
        return payload

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "conversation_id": str(row["conversation_id"]),
            "title": str(row["title"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "message_count": int(row["message_count"]),
        }

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        try:
            metadata = json.loads(str(row["metadata"]))
        except (TypeError, ValueError):
            metadata = {}
        return {
            "message_id": str(row["message_id"]),
            "role": str(row["role"]),
            "kind": str(row["kind"]),
            "content": str(row["content"]),
            "metadata": metadata,
            "created_at": str(row["created_at"]),
        }

    def _owned_conversation(self, connection: sqlite3.Connection, conversation_id: str, user_id: str) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at,
                   (SELECT COUNT(*) FROM chat_messages WHERE conversation_id = c.conversation_id) AS message_count
            FROM chat_conversations AS c
            WHERE conversation_id = ? AND user_id = ? AND deleted_at IS NULL
            """,
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            raise ChatConversationNotFound("Conversation not found")
        return row

    def list_conversations(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT conversation_id, title, created_at, updated_at,
                       (SELECT COUNT(*) FROM chat_messages WHERE conversation_id = c.conversation_id) AS message_count
                FROM chat_conversations AS c
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC LIMIT ?
                """,
                (user_id, bounded_limit),
            ).fetchall()
        return [self._summary(row) for row in rows]

    def get_conversation(self, *, user_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            conversation = self._owned_conversation(connection, conversation_id, user_id)
            rows = connection.execute(
                """
                SELECT message_id, role, kind, content, metadata, created_at
                FROM chat_messages WHERE conversation_id = ? AND user_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id, user_id),
            ).fetchall()
        result = self._summary(conversation)
        result["messages"] = [self._message(row) for row in rows]
        return result

    def append_turn(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        title: str,
        user_message: str,
        assistant_message: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in {"text", "analysis"}:
            raise ChatHistoryError("Unsupported conversation message kind")
        safe_title = self._safe_text(title, limit=MAX_TITLE_LENGTH, label="Conversation title")
        safe_user_message = self._safe_text(user_message, limit=MAX_MESSAGE_LENGTH, label="User message")
        safe_assistant_message = self._safe_text(assistant_message, limit=MAX_MESSAGE_LENGTH, label="Assistant message")
        safe_metadata = self._safe_metadata(metadata)
        conversation_id = conversation_id or f"chat_{uuid.uuid4().hex}"
        timestamp = _now()
        with self._lock, self._connect() as connection:
            if not conversation_id.startswith("chat_"):
                raise ChatHistoryError("Invalid conversation id")
            existing = connection.execute(
                "SELECT conversation_id FROM chat_conversations WHERE conversation_id = ? AND user_id = ? AND deleted_at IS NULL",
                (conversation_id, user_id),
            ).fetchone()
            if existing is None:
                occupied = connection.execute(
                    "SELECT conversation_id FROM chat_conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if occupied is not None:
                    raise ChatConversationNotFound("Conversation not found")
                if conversation_id != conversation_id.strip():
                    raise ChatHistoryError("Invalid conversation id")
                connection.execute(
                    "INSERT INTO chat_conversations(conversation_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, user_id, safe_title, timestamp, timestamp),
                )
            else:
                connection.execute(
                    "UPDATE chat_conversations SET title = ?, updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                    (safe_title, timestamp, conversation_id, user_id),
                )
            connection.executemany(
                "INSERT INTO chat_messages(message_id, conversation_id, user_id, role, kind, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (f"msg_{uuid.uuid4().hex}", conversation_id, user_id, "user", kind, safe_user_message, "{}", timestamp),
                    (f"msg_{uuid.uuid4().hex}", conversation_id, user_id, "assistant", kind, safe_assistant_message, safe_metadata, timestamp),
                ],
            )
        return self.get_conversation(user_id=user_id, conversation_id=conversation_id)

    def delete_conversation(self, *, user_id: str, conversation_id: str) -> None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            self._owned_conversation(connection, conversation_id, user_id)
            connection.execute(
                "UPDATE chat_conversations SET deleted_at = ?, updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                (timestamp, timestamp, conversation_id, user_id),
            )
