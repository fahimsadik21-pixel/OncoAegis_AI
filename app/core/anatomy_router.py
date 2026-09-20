"""Conservative anatomy routing contracts for future imaging models."""

from __future__ import annotations

from dataclasses import dataclass


_BODY_REGION_CANDIDATES: dict[str, tuple[str, ...]] = {
    "CHEST": ("lung",),
    "ABDOMEN": ("liver", "kidney", "pancreas"),
    "PELVIS": ("prostate", "ovary", "colon"),
    "BREAST": ("breast",),
    "NECK": ("thyroid",),
    "HEAD": ("brain",),
}


@dataclass(frozen=True)
class AnatomyRoute:
    """Routing decision without pretending to identify an organ."""

    body_region: str | None
    candidate_organs: tuple[str, ...]
    selected_organ: str | None = None
    route_status: str = "requires_anatomy_model"


def route_body_region(body_region: str | None) -> AnatomyRoute:
    """Return candidate specialist areas for a coarse body-region label.

    BodyPartExamined is not sufficient to select a cancer model. The result
    therefore remains a candidate route until an anatomy/localization model
    confirms the organ.
    """

    normalized = " ".join(
        (body_region or "").upper().replace("_", " ").split()
    ) or None
    candidates = _BODY_REGION_CANDIDATES.get(normalized or "", ())

    return AnatomyRoute(
        body_region=normalized,
        candidate_organs=candidates,
        route_status=(
            "requires_anatomy_model"
            if candidates
            else "unknown_body_region"
        ),
    )
