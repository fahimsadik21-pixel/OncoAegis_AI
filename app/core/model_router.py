"""Conservative routing from an input request to a specialist model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.registry.cancer_registry import get_cancer_capability
from app.registry.model_registry import (
    ModelRegistry,
    get_model_registry,
    normalize_modality,
    normalize_organ,
    normalize_task,
)


def _normalize_cancer_type(value: object) -> str:
    return "_".join(
        str(value or "").strip().lower().replace("-", " ").split()
    )


@dataclass(frozen=True)
class ModelRoute:
    """A model-selection result with an explicit safe fallback state."""

    modality: str
    organ: str | None
    task: str | None
    cancer_type: str | None
    selected_model_id: str | None
    candidate_model_ids: tuple[str, ...]
    route_status: str
    reason: str
    safety_status: str
    malignancy_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate_model_ids"] = list(self.candidate_model_ids)
        return value


def route_specialist_model(
    *,
    modality: object,
    organ: object | None = None,
    task: object | None = None,
    cancer_type: object | None = None,
    require_available: bool = True,
    registry: ModelRegistry | None = None,
) -> ModelRoute:
    """Select a registered model without guessing unsupported anatomy.

    A request with multiple matching models must include a task. A cancer area
    with no registered model returns an explicit unavailable status.
    """

    normalized_modality = normalize_modality(modality)
    normalized_organ = normalize_organ(organ) if organ else None
    normalized_task = normalize_task(task) if task else None
    normalized_cancer = (
        _normalize_cancer_type(cancer_type) if cancer_type else None
    )
    empty_candidates: tuple[str, ...] = ()

    if not normalized_modality:
        return ModelRoute(
            modality="",
            organ=normalized_organ,
            task=normalized_task,
            cancer_type=normalized_cancer,
            selected_model_id=None,
            candidate_model_ids=empty_candidates,
            route_status="invalid_route_request",
            reason="A modality is required before model routing.",
            safety_status="no_model_selected",
        )

    capability = (
        get_cancer_capability(normalized_cancer)
        if normalized_cancer
        else None
    )
    if normalized_cancer and capability is None:
        return ModelRoute(
            modality=normalized_modality,
            organ=normalized_organ,
            task=normalized_task,
            cancer_type=normalized_cancer,
            selected_model_id=None,
            candidate_model_ids=empty_candidates,
            route_status="unknown_cancer_type",
            reason="The requested cancer area is not registered.",
            safety_status="no_model_selected",
        )

    if capability is not None:
        allowed_modalities = {
            normalize_modality(value) for value in capability["modalities"]
        }
        allowed_organs = {
            normalize_organ(value) for value in capability["organs"]
        }
        if normalized_modality not in allowed_modalities:
            return ModelRoute(
                modality=normalized_modality,
                organ=normalized_organ,
                task=normalized_task,
                cancer_type=normalized_cancer,
                selected_model_id=None,
                candidate_model_ids=empty_candidates,
                route_status="unsupported_cancer_modality",
                reason="The cancer registry does not support this modality.",
                safety_status="no_model_selected",
            )
        if normalized_organ and normalized_organ not in allowed_organs:
            return ModelRoute(
                modality=normalized_modality,
                organ=normalized_organ,
                task=normalized_task,
                cancer_type=normalized_cancer,
                selected_model_id=None,
                candidate_model_ids=empty_candidates,
                route_status="unsupported_cancer_organ",
                reason="The cancer registry does not support this organ.",
                safety_status="no_model_selected",
            )

    model_registry = registry or get_model_registry()
    candidates = model_registry.find(
        modality=normalized_modality,
        organ=normalized_organ,
        task=normalized_task,
        available_only=require_available,
    )
    if capability is not None:
        allowed_ids = set(capability["models"])
        candidates = tuple(
            candidate for candidate in candidates
            if candidate.model_id in allowed_ids
        )

    candidate_ids = tuple(candidate.model_id for candidate in candidates)
    if not candidates:
        if capability is not None and not capability["models"]:
            status = "specialist_model_unavailable"
            reason = "No specialist model is registered for this capability."
        else:
            status = "checkpoint_missing" if require_available else "no_matching_model"
            reason = (
                "A matching model is registered but its checkpoint is unavailable."
                if status == "checkpoint_missing"
                else "No registered model matches the requested route."
            )
        return ModelRoute(
            modality=normalized_modality,
            organ=normalized_organ,
            task=normalized_task,
            cancer_type=normalized_cancer,
            selected_model_id=None,
            candidate_model_ids=empty_candidates,
            route_status=status,
            reason=reason,
            safety_status="no_model_selected",
        )

    if len(candidates) > 1 and normalized_task is None:
        return ModelRoute(
            modality=normalized_modality,
            organ=normalized_organ,
            task=None,
            cancer_type=normalized_cancer,
            selected_model_id=None,
            candidate_model_ids=candidate_ids,
            route_status="task_required",
            reason="Multiple specialist models match; specify detection, segmentation or classification.",
            safety_status="no_model_selected",
        )

    top_priority = candidates[0].priority
    tied = tuple(
        candidate for candidate in candidates
        if candidate.priority == top_priority
    )
    if len(tied) > 1:
        return ModelRoute(
            modality=normalized_modality,
            organ=normalized_organ,
            task=normalized_task,
            cancer_type=normalized_cancer,
            selected_model_id=None,
            candidate_model_ids=candidate_ids,
            route_status="ambiguous_model_route",
            reason="Matching specialist models have equal routing priority.",
            safety_status="no_model_selected",
        )

    selected = candidates[0]
    return ModelRoute(
        modality=normalized_modality,
        organ=normalized_organ,
        task=normalized_task,
        cancer_type=normalized_cancer,
        selected_model_id=selected.model_id,
        candidate_model_ids=candidate_ids,
        route_status="model_selected",
        reason=(
            "Registered research model selected; its output requires expert "
            "review and cannot confirm malignancy."
        ),
        safety_status="research_model_requires_expert_review",
        malignancy_confirmation=False,
    )


def route_model(**kwargs: object) -> ModelRoute:
    """Short alias for callers that use the generic router name."""

    return route_specialist_model(**kwargs)
