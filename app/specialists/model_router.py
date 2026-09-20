"""
OncoAegis specialist model routing engine.

Selects the best available specialist model
based on modality, organ and task requirements.
"""

from __future__ import annotations

from typing import Any

from app.registry.model_registry import (
    get_model_registry,
)

from app.registry.dataset_model_map import (
    get_models_for_dataset,
)

from app.registry.dataset_validator import (
    dataset_validator,
)


class ModelRouter:
    """
    Routes clinical imaging requests
    to suitable specialist models.
    """

    def __init__(self):
        self.model_registry = get_model_registry()
        self.validator = dataset_validator


    def route(
        self,
        *,
        modality: str,
        organ: str | None = None,
        task: str | None = None,
        require_available: bool = True,
    ) -> dict[str, Any]:

        candidates = self.model_registry.find(
            modality=modality,
            organ=organ,
            task=task,
            available_only=require_available,
        )

        if not candidates:

            return {
                "selected_model_id": None,
                "candidate_model_ids": [],
                "route_status": "no_match",
                "reason": (
                    "No specialist model matched "
                    "the requested capability"
                ),
                "safety_status": "no_model_selected",
            }


        selected = candidates[0]


        dataset_status = self.validator.validate_dataset(
            selected.dataset
        )


        if not dataset_status.get("ready"):

            return {
                "selected_model_id": None,
                "candidate_model_ids": [
                    m.model_id for m in candidates
                ],
                "route_status": "dataset_unavailable",
                "reason": (
                    f"Dataset {selected.dataset} "
                    "is not ready"
                ),
                "safety_status": "blocked",
            }


        return {
            "selected_model_id": selected.model_id,
            "candidate_model_ids": [
                m.model_id for m in candidates
            ],
            "route_status": "selected",
            "reason": (
                f"Selected {selected.name} "
                f"using {selected.dataset} dataset"
            ),
            "safety_status": (
                "research_model"
            ),
        }


model_router = ModelRouter()