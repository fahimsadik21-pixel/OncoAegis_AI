from __future__ import annotations

import gc
import logging
from typing import Any

from app.deployment.checkpoint_bootstrap import ensure_model_checkpoint
from app.core.specialist_factory import create_specialist


_SPECIALISTS: dict[str, Any] = {}
LOGGER = logging.getLogger(__name__)


def get_or_create_specialist(
    model_id: str
):

    if model_id in _SPECIALISTS:
        return _SPECIALISTS[model_id]


    ensure_model_checkpoint(model_id)

    specialist = create_specialist(
        model_id
    )


    _SPECIALISTS[model_id] = specialist


    return specialist



def get_specialist(
    model_id: str
):

    specialist = _SPECIALISTS.get(
        model_id
    )

    if specialist is None:
        raise ValueError(
            f"Specialist not loaded: {model_id}"
        )

    return specialist



def list_specialists():

    return list(
        _SPECIALISTS.keys()
    )


def release_specialist(model_id: str) -> bool:
    """Drop one cached specialist and release its model graph when possible."""

    specialist = _SPECIALISTS.pop(model_id, None)
    if specialist is None:
        return False

    for cleanup_name in ("close", "unload", "cleanup"):
        cleanup = getattr(specialist, cleanup_name, None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:  # Cleanup must never hide an analysis result.
                LOGGER.debug(
                    "Specialist cleanup failed for %s via %s",
                    model_id,
                    cleanup_name,
                    exc_info=True,
                )
            break

    del specialist
    gc.collect()
    return True


def clear_specialists() -> int:
    """Release every cached specialist; useful for controlled shutdown/tests."""

    released = 0
    for model_id in tuple(_SPECIALISTS):
        released += int(release_specialist(model_id))
    return released
