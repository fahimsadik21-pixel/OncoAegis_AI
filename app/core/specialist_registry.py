from __future__ import annotations

from typing import Any

from app.core.specialist_factory import create_specialist


_SPECIALISTS: dict[str, Any] = {}


def get_or_create_specialist(
    model_id: str
):

    if model_id in _SPECIALISTS:
        return _SPECIALISTS[model_id]


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
