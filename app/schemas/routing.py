"""API contracts for specialist-model routing."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelRouteRequest(BaseModel):
    modality: str = Field(min_length=1)
    organ: str | None = None
    task: str | None = None
    cancer_type: str | None = None
    require_available: bool = True


class ModelRouteResponse(BaseModel):
    modality: str
    organ: str | None = None
    task: str | None = None
    cancer_type: str | None = None
    selected_model_id: str | None = None
    candidate_model_ids: tuple[str, ...] = ()
    route_status: str
    reason: str
    safety_status: str
    malignancy_confirmation: bool = False
