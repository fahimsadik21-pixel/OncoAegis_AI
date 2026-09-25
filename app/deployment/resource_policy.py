"""Small hosting-aware policy helpers for constrained model deployments."""

from __future__ import annotations

import os
from pathlib import Path

from app.registry.model_registry import ModelSpec


def _enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def model_eviction_enabled() -> bool:
    """Whether a loaded specialist should be released after each request.

    Railway and Render default to this conservative mode. A local workstation
    keeps models warm unless the environment variable is explicitly set.
    """

    configured = os.getenv("ONCOAEGIS_EVICT_MODELS_AFTER_REQUEST", "").strip()
    if configured:
        return _enabled(configured)
    return any(
        os.getenv(marker, "").strip()
        for marker in ("RENDER", "RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID")
    )


def hosted_memory_limit_mb() -> int | None:
    """Return a container memory limit when the host exposes one.

    Railway uses Linux cgroups.  The helper is deliberately best-effort so
    local Windows development keeps its existing behaviour.
    """

    configured = os.getenv("ONCOAEGIS_MEMORY_LIMIT_MB", "").strip()
    if configured:
        try:
            value = int(configured)
        except ValueError:
            value = 0
        if value > 0:
            return value

    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for candidate in candidates:
        try:
            raw_value = candidate.read_text(encoding="utf-8").strip()
            if raw_value == "max":
                continue
            value = int(raw_value)
        except (OSError, ValueError):
            continue
        # cgroup uses an enormous sentinel for an effectively unlimited host.
        if value <= 0 or value >= 1 << 50:
            continue
        return max(1, value // (1024 * 1024))
    return None


def required_memory_mb(spec: ModelSpec) -> int | None:
    """Return a conservative hosted-memory floor for heavy research models."""

    if "3D" not in spec.input_type.upper() and "3D" not in spec.model_kind.upper():
        return None
    if "Two-stage" in spec.model_kind:
        return 2048
    return 1536


def hosted_memory_error(spec: ModelSpec) -> str | None:
    """Prevent a hosted 3D request from killing a constrained web process."""

    if not any(
        os.getenv(marker, "").strip()
        for marker in ("RENDER", "RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID")
    ):
        return None
    required = required_memory_mb(spec)
    limit = hosted_memory_limit_mb()
    if required is None or limit is None or limit >= required:
        return None
    return (
        f"{spec.name} needs at least {required} MB of service memory for one "
        f"research 3D analysis, but this Railway deployment has {limit} MB. "
        "Increase the service memory, then try the same original CT/MRI volume again."
    )
