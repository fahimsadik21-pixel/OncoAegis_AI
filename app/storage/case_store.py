"""SQLite-backed case, result and audit storage.

The storage boundary deliberately keeps raw uploaded medical files out of the
database.  It stores normalized analysis results, privacy-safe metadata,
cryptographic input/result hashes and an append-only tamper-evident audit
chain.  The default database path is configurable through
``ONCOAEGIS_DATABASE_PATH``.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Iterable, Iterator
import uuid

from app.schemas.analysis_result import StandardAnalysisResult


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = _PROJECT_ROOT / "outputs" / "oncoaegis.sqlite3"
DEFAULT_RETENTION_DAYS = 30
MAX_RETENTION_DAYS = 3650
MAX_RESULT_JSON_BYTES = 32 * 1024 * 1024
SCHEMA_VERSION = "1"
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StorageError(RuntimeError):
    """Base class for safe persistence failures."""


class CaseNotFoundError(StorageError):
    """The requested case or result does not exist."""


class CaseDeletedError(StorageError):
    """The requested case has already been deleted or purged."""


class InvalidCaseIDError(StorageError):
    """A caller supplied an unsafe or invalid case identifier."""


class StorageConflictError(StorageError):
    """A result identifier would overwrite an existing stored result."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StorageError("Value cannot be serialized safely for storage") from exc


def _parse_retention_days(value: object | None) -> int:
    raw = value
    if raw is None or raw == "":
        raw = os.getenv("ONCOAEGIS_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    try:
        days = int(raw)
    except (TypeError, ValueError) as exc:
        raise StorageError("Retention days must be an integer") from exc
    if days < 1 or days > MAX_RETENTION_DAYS:
        raise StorageError(
            f"Retention days must be between 1 and {MAX_RETENTION_DAYS}"
        )
    return days


def normalize_actor_id(actor_id: object | None) -> str:
    """Return a bounded, control-character-free audit actor label."""

    if not isinstance(actor_id, str):
        return "anonymous"
    value = actor_id.strip()
    if not value or len(value) > 128 or any(ord(char) < 32 for char in value):
        return "anonymous"
    return value


def validate_case_id(case_id: object | None) -> str:
    """Validate a caller-provided ID or create a new opaque case ID."""

    if case_id is None or not str(case_id).strip():
        return f"case-{uuid.uuid4()}"
    value = str(case_id).strip()
    if not CASE_ID_PATTERN.fullmatch(value):
        raise InvalidCaseIDError(
            "case_id must be 1-128 characters using letters, numbers, '.', '_' or '-'"
        )
    return value


def compute_input_hash(files: Iterable[object]) -> str:
    """Hash an ordered upload payload without retaining its filenames.

    File extensions are included as format hints, while complete filenames and
    their possible patient identifiers are deliberately excluded.
    """

    digest = hashlib.sha256()
    entries = list(files)
    digest.update(b"oncoaegis-input-v1\0")
    digest.update(len(entries).to_bytes(8, "big"))
    for entry in entries:
        name = str(getattr(entry, "name", "") or "").lower()
        suffix = ".nii.gz" if name.endswith(".nii.gz") else Path(name).suffix.lower()
        suffix_bytes = suffix.encode("utf-8")
        content = bytes(getattr(entry, "content", b""))
        digest.update(len(suffix_bytes).to_bytes(4, "big"))
        digest.update(suffix_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _checkpoint_hash(checkpoint: object | None) -> str | None:
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    try:
        path = path.resolve()
        path.relative_to(_PROJECT_ROOT.resolve())
    except (OSError, ValueError):
        return None
    return _sha256_file(path)


def _merge_context(existing: str | None, incoming: object | None, multiple: str) -> str | None:
    value = str(incoming).strip() if incoming is not None else ""
    if not value:
        return existing
    if not existing or existing == value:
        return value
    return multiple


class CaseStore:
    """Atomic SQLite persistence for cases, results and audit events."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        retention_days: int | None = None,
    ) -> None:
        configured_path = db_path
        if configured_path is None:
            configured_path = os.getenv(
                "ONCOAEGIS_DATABASE_PATH",
                str(DEFAULT_DATABASE_PATH),
            )
        self._memory = str(configured_path) == ":memory:"
        self.db_path = None if self._memory else Path(configured_path).expanduser()
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = _parse_retention_days(retention_days)
        self._memory_connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        if self._memory:
            self._memory_connection = sqlite3.connect(
                ":memory:",
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            self._configure_connection(self._memory_connection)
        self.initialize()

    @property
    def database_label(self) -> str:
        return ":memory:" if self._memory else str(self.db_path)

    @staticmethod
    def _configure_connection(connection: sqlite3.Connection) -> None:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = NORMAL")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            yield self._memory_connection
            return
        if self.db_path is None:
            raise StorageError("Database path is not configured")
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            isolation_level=None,
        )
        try:
            self._configure_connection(connection)
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                except Exception:
                    connection.rollback()
                    raise
                else:
                    connection.commit()

    def initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'deleted')),
                    modality TEXT,
                    organ TEXT,
                    retention_expires_at TEXT NOT NULL,
                    deleted_at TEXT,
                    deletion_reason TEXT,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    data_purged INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    analysis_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    checkpoint_hash TEXT,
                    input_hash TEXT NOT NULL,
                    input_hash_algorithm TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases(case_id)
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    case_id TEXT,
                    analysis_id TEXT,
                    actor_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status_updated
                    ON cases(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cases_retention
                    ON cases(status, retention_expires_at);
                CREATE INDEX IF NOT EXISTS idx_results_case_created
                    ON analysis_results(case_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_case_sequence
                    ON audit_events(case_id, sequence DESC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(cases)").fetchall()
            }
            if "owner_user_id" not in columns:
                connection.execute("ALTER TABLE cases ADD COLUMN owner_user_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cases_owner_status_updated "
                "ON cases(owner_user_id, status, updated_at DESC)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )

    @staticmethod
    def _case_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "case_id": row["case_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "modality": row["modality"],
            "organ": row["organ"],
            "retention_expires_at": row["retention_expires_at"],
            "deleted_at": row["deleted_at"],
            "deletion_reason": row["deletion_reason"],
            "result_count": int(row["result_count"]),
            "data_purged": bool(row["data_purged"]),
        }

    @staticmethod
    def _result_summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "analysis_id": row["analysis_id"],
            "case_id": row["case_id"],
            "created_at": row["created_at"],
            "model_id": row["model_id"],
            "model_version": row["model_version"],
            "checkpoint_hash": row["checkpoint_hash"],
            "input_hash": row["input_hash"],
            "input_hash_algorithm": row["input_hash_algorithm"],
            "result_hash": row["result_hash"],
        }

    @staticmethod
    def _audit_hash_payload(
        *,
        sequence: int,
        event_id: str,
        occurred_at: str,
        case_id: str | None,
        analysis_id: str | None,
        actor_id: str,
        action: str,
        object_type: str,
        object_id: str,
        details_json: str,
        previous_event_hash: str | None,
    ) -> dict[str, Any]:
        return {
            "sequence": sequence,
            "event_id": event_id,
            "occurred_at": occurred_at,
            "case_id": case_id,
            "analysis_id": analysis_id,
            "actor_id": actor_id,
            "action": action,
            "object_type": object_type,
            "object_id": object_id,
            "details_json": details_json,
            "previous_event_hash": previous_event_hash,
        }

    def _append_audit(
        self,
        connection: sqlite3.Connection,
        *,
        action: str,
        object_type: str,
        object_id: str,
        actor_id: object | None,
        case_id: str | None = None,
        analysis_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else None
        event_id = str(uuid.uuid4())
        occurred_at = _utc_now()
        safe_actor = normalize_actor_id(actor_id)
        details_json = _canonical_json(details or {})
        cursor = connection.execute(
            """
            INSERT INTO audit_events(
                event_id, occurred_at, case_id, analysis_id, actor_id,
                action, object_type, object_id, details_json,
                previous_event_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                occurred_at,
                case_id,
                analysis_id,
                safe_actor,
                action,
                object_type,
                object_id,
                details_json,
                previous_hash,
                "pending",
            ),
        )
        sequence = int(cursor.lastrowid)
        payload = self._audit_hash_payload(
            sequence=sequence,
            event_id=event_id,
            occurred_at=occurred_at,
            case_id=case_id,
            analysis_id=analysis_id,
            actor_id=safe_actor,
            action=action,
            object_type=object_type,
            object_id=object_id,
            details_json=details_json,
            previous_event_hash=previous_hash,
        )
        event_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        connection.execute(
            "UPDATE audit_events SET event_hash = ? WHERE sequence = ?",
            (event_hash, sequence),
        )
        return {
            "sequence": sequence,
            "event_id": event_id,
            "occurred_at": occurred_at,
            "case_id": case_id,
            "analysis_id": analysis_id,
            "actor_id": safe_actor,
            "action": action,
            "object_type": object_type,
            "object_id": object_id,
            "details": details or {},
            "previous_event_hash": previous_hash,
            "event_hash": event_hash,
        }

    def _ensure_case(
        self,
        connection: sqlite3.Connection,
        *,
        case_id: str,
        modality: object | None,
        organ: object | None,
        now: str,
        owner_user_id: object | None = None,
    ) -> tuple[sqlite3.Row, bool]:
        row = connection.execute(
            "SELECT * FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is not None:
            if owner_user_id is not None and row["owner_user_id"] not in (None, normalize_actor_id(owner_user_id)):
                raise CaseNotFoundError(f"Case not found: {case_id}")
            if row["status"] == "deleted":
                raise CaseDeletedError(f"Case has been deleted: {case_id}")
            merged_modality = _merge_context(row["modality"], modality, "MULTIMODAL")
            merged_organ = _merge_context(row["organ"], organ, "multiple")
            if merged_modality != row["modality"] or merged_organ != row["organ"]:
                connection.execute(
                    """
                    UPDATE cases
                    SET modality = ?, organ = ?, updated_at = ?
                    WHERE case_id = ?
                    """,
                    (merged_modality, merged_organ, now, case_id),
                )
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?",
                    (case_id,),
                ).fetchone()
            return row, False

        expires = (
            datetime.fromisoformat(now) + timedelta(days=self.retention_days)
        ).isoformat()
        connection.execute(
            """
            INSERT INTO cases(
                case_id, created_at, updated_at, status, modality, organ,
                retention_expires_at, result_count, data_purged, owner_user_id
            ) VALUES (?, ?, ?, 'active', ?, ?, ?, 0, 0, ?)
            """,
            (
                case_id,
                now,
                now,
                str(modality).strip() if modality else None,
                str(organ).strip() if organ else None,
                expires,
                normalize_actor_id(owner_user_id) if owner_user_id is not None else None,
            ),
        )
        row = connection.execute(
            "SELECT * FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise StorageError("Case could not be created")
        return row, True

    def store_result(
        self,
        result: StandardAnalysisResult,
        *,
        files: Iterable[object],
        actor_id: object | None = None,
        case_id: object | None = None,
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        """Persist one result atomically and return safe storage metadata."""

        if not isinstance(result, StandardAnalysisResult):
            raise StorageError("Only StandardAnalysisResult can be persisted")
        resolved_case_id = validate_case_id(case_id or result.case_id)
        input_hash = compute_input_hash(files)
        result.case_id = resolved_case_id
        result.provenance["input_hash"] = input_hash
        result.provenance["input_hash_algorithm"] = "sha256"
        result.provenance["storage"] = {
            "backend": "sqlite",
            "raw_input_persisted": False,
        }
        payload = result.model_dump(mode="json")
        result_json = _canonical_json(payload)
        if len(result_json.encode("utf-8")) > MAX_RESULT_JSON_BYTES:
            raise StorageError("Analysis result exceeds the storage size limit")
        result_hash = hashlib.sha256(result_json.encode("utf-8")).hexdigest()

        specialist = result.specialist
        model_id = str(
            getattr(specialist, "model_id", None)
            or result.provenance.get("execution", {}).get("model_id")
            or "unknown"
        )
        model_version = str(getattr(specialist, "model_version", None) or "unknown")
        checkpoint_hash = _checkpoint_hash(getattr(specialist, "checkpoint", None))
        analysis_id = str(result.analysis_id)
        now = _utc_now()
        modality = result.input.get("modality") if isinstance(result.input, dict) else None
        organ = result.input.get("organ") if isinstance(result.input, dict) else None

        with self._transaction() as connection:
            try:
                case_row, created = self._ensure_case(
                    connection,
                    case_id=resolved_case_id,
                    modality=modality,
                    organ=organ,
                    now=now,
                    owner_user_id=owner_user_id or actor_id,
                )
                existing = connection.execute(
                    "SELECT 1 FROM analysis_results WHERE analysis_id = ?",
                    (analysis_id,),
                ).fetchone()
                if existing is not None:
                    raise StorageConflictError(
                        f"Analysis result already exists: {analysis_id}"
                    )
                connection.execute(
                    """
                    INSERT INTO analysis_results(
                        analysis_id, case_id, created_at, model_id, model_version,
                        checkpoint_hash, input_hash, input_hash_algorithm,
                        result_hash, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        resolved_case_id,
                        now,
                        model_id,
                        model_version,
                        checkpoint_hash,
                        input_hash,
                        "sha256",
                        result_hash,
                        result_json,
                    ),
                )
                expires = (
                    datetime.fromisoformat(now)
                    + timedelta(days=self.retention_days)
                ).isoformat()
                connection.execute(
                    """
                    UPDATE cases
                    SET updated_at = ?, retention_expires_at = ?,
                        result_count = result_count + 1
                    WHERE case_id = ?
                    """,
                    (now, expires, resolved_case_id),
                )
                if created:
                    self._append_audit(
                        connection,
                        action="case.created",
                        object_type="case",
                        object_id=resolved_case_id,
                        case_id=resolved_case_id,
                        actor_id=actor_id,
                        details={
                            "modality": modality,
                            "organ": organ,
                            "retention_expires_at": expires,
                        },
                    )
                self._append_audit(
                    connection,
                    action="result.stored",
                    object_type="analysis_result",
                    object_id=analysis_id,
                    case_id=resolved_case_id,
                    analysis_id=analysis_id,
                    actor_id=actor_id,
                    details={
                        "model_id": model_id,
                        "model_version": model_version,
                        "checkpoint_hash": checkpoint_hash,
                        "input_hash": input_hash,
                        "result_hash": result_hash,
                    },
                )
            except sqlite3.IntegrityError as exc:
                raise StorageConflictError("Could not persist analysis result") from exc

            stored_case = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?",
                (resolved_case_id,),
            ).fetchone()
            stored_result = connection.execute(
                "SELECT * FROM analysis_results WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
            if stored_case is None or stored_result is None:
                raise StorageError("Stored analysis could not be reloaded")
            return {
                "case": self._case_dict(stored_case),
                "result": self._result_summary(stored_result),
            }

    def list_cases(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
        actor_id: object | None = None,
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        with self._transaction() as connection:
            clauses = [] if include_deleted else ["status = 'active'"]
            params: list[object] = []
            if owner_user_id is not None:
                clauses.append("owner_user_id = ?")
                params.append(normalize_actor_id(owner_user_id))
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = connection.execute(
                f"""
                SELECT * FROM cases {where}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            total = connection.execute(
                f"SELECT COUNT(*) AS count FROM cases {where}", tuple(params)
            ).fetchone()["count"]
            self._append_audit(
                connection,
                action="cases.listed",
                object_type="case_collection",
                object_id="cases",
                actor_id=actor_id,
                details={
                    "limit": limit,
                    "offset": offset,
                    "include_deleted": bool(include_deleted),
                    "returned": len(rows),
                },
            )
            return {
                "cases": [self._case_dict(row) for row in rows],
                "total": int(total),
                "limit": limit,
                "offset": offset,
            }

    def get_case(
        self,
        case_id: object,
        *,
        include_results: bool = True,
        actor_id: object | None = None,
        include_deleted: bool = False,
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        resolved_case_id = validate_case_id(case_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?",
                (resolved_case_id,),
            ).fetchone()
            if row is None:
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if owner_user_id is not None and row["owner_user_id"] != normalize_actor_id(owner_user_id):
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if row["status"] == "deleted" and not include_deleted:
                raise CaseDeletedError(f"Case has been deleted: {resolved_case_id}")
            response: dict[str, Any] = {"case": self._case_dict(row)}
            if include_results:
                result_rows = connection.execute(
                    """
                    SELECT * FROM analysis_results
                    WHERE case_id = ?
                    ORDER BY created_at DESC
                    """,
                    (resolved_case_id,),
                ).fetchall()
                response["results"] = [
                    self._result_summary(result) for result in result_rows
                ]
            self._append_audit(
                connection,
                action="case.viewed",
                object_type="case",
                object_id=resolved_case_id,
                case_id=resolved_case_id,
                actor_id=actor_id,
                details={"include_results": bool(include_results)},
            )
            return response

    def list_results(
        self,
        case_id: object,
        *,
        actor_id: object | None = None,
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        resolved_case_id = validate_case_id(case_id)
        with self._transaction() as connection:
            case = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?",
                (resolved_case_id,),
            ).fetchone()
            if case is None:
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if owner_user_id is not None and case["owner_user_id"] != normalize_actor_id(owner_user_id):
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if case["status"] == "deleted":
                raise CaseDeletedError(f"Case has been deleted: {resolved_case_id}")
            rows = connection.execute(
                """
                SELECT * FROM analysis_results
                WHERE case_id = ?
                ORDER BY created_at DESC
                """,
                (resolved_case_id,),
            ).fetchall()
            self._append_audit(
                connection,
                action="results.listed",
                object_type="case_results",
                object_id=resolved_case_id,
                case_id=resolved_case_id,
                actor_id=actor_id,
                details={"returned": len(rows)},
            )
            return {
                "case_id": resolved_case_id,
                "results": [self._result_summary(row) for row in rows],
            }

    def get_result(
        self,
        case_id: object,
        analysis_id: object,
        *,
        actor_id: object | None = None,
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        resolved_case_id = validate_case_id(case_id)
        resolved_analysis_id = str(analysis_id).strip()
        if not resolved_analysis_id or len(resolved_analysis_id) > 128:
            raise CaseNotFoundError("Analysis result not found")
        with self._transaction() as connection:
            case = connection.execute(
                "SELECT status FROM cases WHERE case_id = ?",
                (resolved_case_id,),
            ).fetchone()
            if case is None:
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if owner_user_id is not None and case["owner_user_id"] != normalize_actor_id(owner_user_id):
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if case["status"] == "deleted":
                raise CaseDeletedError(f"Case has been deleted: {resolved_case_id}")
            row = connection.execute(
                """
                SELECT * FROM analysis_results
                WHERE case_id = ? AND analysis_id = ?
                """,
                (resolved_case_id, resolved_analysis_id),
            ).fetchone()
            if row is None:
                raise CaseNotFoundError("Analysis result not found")
            try:
                result_payload = json.loads(row["result_json"])
            except json.JSONDecodeError as exc:
                raise StorageError("Stored analysis result is invalid JSON") from exc
            self._append_audit(
                connection,
                action="result.viewed",
                object_type="analysis_result",
                object_id=resolved_analysis_id,
                case_id=resolved_case_id,
                analysis_id=resolved_analysis_id,
                actor_id=actor_id,
                details={"result_hash": row["result_hash"]},
            )
            return {
                "storage": self._result_summary(row),
                "result": result_payload,
            }

    def list_audit(
        self,
        case_id: object,
        *,
        limit: int = 100,
        actor_id: object | None = None,
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        resolved_case_id = validate_case_id(case_id)
        limit = max(1, min(int(limit), 500))
        with self._transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM cases WHERE case_id = ?",
                (resolved_case_id,),
            ).fetchone()
            if exists is None:
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if owner_user_id is not None:
                owned = connection.execute(
                    "SELECT 1 FROM cases WHERE case_id = ? AND owner_user_id = ?",
                    (resolved_case_id, normalize_actor_id(owner_user_id)),
                ).fetchone()
                if owned is None:
                    raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                WHERE case_id = ?
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (resolved_case_id, limit),
            ).fetchall()
            response = [self._audit_dict(row) for row in rows]
            self._append_audit(
                connection,
                action="audit.viewed",
                object_type="case_audit",
                object_id=resolved_case_id,
                case_id=resolved_case_id,
                actor_id=actor_id,
                details={"limit": limit, "returned": len(response)},
            )
            return {"case_id": resolved_case_id, "events": response}

    @staticmethod
    def _audit_dict(row: sqlite3.Row) -> dict[str, Any]:
        try:
            details = json.loads(row["details_json"])
        except json.JSONDecodeError:
            details = {"invalid_details": True}
        return {
            "sequence": int(row["sequence"]),
            "event_id": row["event_id"],
            "occurred_at": row["occurred_at"],
            "case_id": row["case_id"],
            "analysis_id": row["analysis_id"],
            "actor_id": row["actor_id"],
            "action": row["action"],
            "object_type": row["object_type"],
            "object_id": row["object_id"],
            "details": details,
            "previous_event_hash": row["previous_event_hash"],
            "event_hash": row["event_hash"],
        }

    def _purge_case_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        actor_id: object | None,
        reason: str,
        action: str,
    ) -> dict[str, Any]:
        case_id = row["case_id"]
        result_count = int(
            connection.execute(
                "SELECT COUNT(*) AS count FROM analysis_results WHERE case_id = ?",
                (case_id,),
            ).fetchone()["count"]
        )
        now = _utc_now()
        self._append_audit(
            connection,
            action=action,
            object_type="case",
            object_id=case_id,
            case_id=case_id,
            actor_id=actor_id,
            details={
                "reason": reason[:500],
                "purged_result_count": result_count,
                "deletion_mode": "payload_purge_with_tombstone",
            },
        )
        connection.execute(
            "DELETE FROM analysis_results WHERE case_id = ?",
            (case_id,),
        )
        connection.execute(
            """
            UPDATE cases
            SET status = 'deleted', updated_at = ?, deleted_at = ?,
                deletion_reason = ?, result_count = 0, data_purged = 1
            WHERE case_id = ?
            """,
            (now, now, reason[:500], case_id),
        )
        updated = connection.execute(
            "SELECT * FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if updated is None:
            raise StorageError("Deleted case tombstone could not be reloaded")
        return self._case_dict(updated)

    def delete_case(
        self,
        case_id: object,
        *,
        actor_id: object | None = None,
        reason: str = "user_requested_deletion",
        owner_user_id: object | None = None,
    ) -> dict[str, Any]:
        resolved_case_id = validate_case_id(case_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?",
                (resolved_case_id,),
            ).fetchone()
            if row is None:
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if owner_user_id is not None and row["owner_user_id"] != normalize_actor_id(owner_user_id):
                raise CaseNotFoundError(f"Case not found: {resolved_case_id}")
            if row["status"] == "deleted":
                raise CaseDeletedError(f"Case has already been deleted: {resolved_case_id}")
            return self._purge_case_in_transaction(
                connection,
                row=row,
                actor_id=actor_id,
                reason=reason,
                action="case.deleted",
            )

    def purge_expired_cases(
        self,
        *,
        actor_id: object | None = "retention-job",
        now: str | None = None,
    ) -> dict[str, Any]:
        cutoff = now or _utc_now()
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cases
                WHERE status = 'active' AND retention_expires_at <= ?
                ORDER BY retention_expires_at ASC
                """,
                (cutoff,),
            ).fetchall()
            purged = []
            for row in rows:
                purged.append(
                    self._purge_case_in_transaction(
                        connection,
                        row=row,
                        actor_id=actor_id,
                        reason="retention_expired",
                        action="retention.purged",
                    )["case_id"]
                )
            return {
                "purged_case_ids": purged,
                "purged_count": len(purged),
                "evaluated_at": cutoff,
            }

    def verify_audit_chain(self) -> dict[str, Any]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY sequence ASC"
            ).fetchall()
        previous_hash: str | None = None
        for row in rows:
            if row["previous_event_hash"] != previous_hash:
                return {
                    "valid": False,
                    "checked_events": int(row["sequence"]),
                    "first_invalid_sequence": int(row["sequence"]),
                    "reason": "previous_event_hash_mismatch",
                }
            payload = self._audit_hash_payload(
                sequence=int(row["sequence"]),
                event_id=row["event_id"],
                occurred_at=row["occurred_at"],
                case_id=row["case_id"],
                analysis_id=row["analysis_id"],
                actor_id=row["actor_id"],
                action=row["action"],
                object_type=row["object_type"],
                object_id=row["object_id"],
                details_json=row["details_json"],
                previous_event_hash=row["previous_event_hash"],
            )
            expected_hash = hashlib.sha256(
                _canonical_json(payload).encode("utf-8")
            ).hexdigest()
            if row["event_hash"] != expected_hash:
                return {
                    "valid": False,
                    "checked_events": int(row["sequence"]),
                    "first_invalid_sequence": int(row["sequence"]),
                    "reason": "event_hash_mismatch",
                }
            previous_hash = row["event_hash"]
        return {
            "valid": True,
            "checked_events": len(rows),
            "first_invalid_sequence": None,
            "reason": None,
        }

    def policy(self) -> dict[str, Any]:
        return {
            "backend": "sqlite",
            "retention_days": self.retention_days,
            "raw_inputs_stored": False,
            "result_payloads_stored": True,
            "deletion_mode": "payload_purge_with_case_tombstone",
            "audit_chain": "sha256_append_only",
            "audit_events_retained_after_case_deletion": True,
            "database_path_configured": bool(self.db_path is not None),
        }


_default_store: CaseStore | None = None
_default_store_lock = threading.Lock()


def get_case_store() -> CaseStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = CaseStore()
    return _default_store


def set_case_store(store: CaseStore | None) -> None:
    """Replace the process store, primarily for isolated tests/deployments."""

    global _default_store
    with _default_store_lock:
        _default_store = store
