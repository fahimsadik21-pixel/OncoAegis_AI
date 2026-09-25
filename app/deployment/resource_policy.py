"""Small hosting-aware policy helpers for constrained model deployments."""

from __future__ import annotations

import os


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
