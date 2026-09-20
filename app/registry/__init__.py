"""Registries for datasets, specialist models and cancer capabilities."""

from app.registry.cancer_registry import (
    CANCER_REGISTRY,
    get_cancer_capabilities,
    get_cancer_capability,
    get_models_for_cancer,
)
from app.registry.model_registry import (
    MODEL_REGISTRY,
    ModelRegistry,
    ModelSpec,
    get_all_models,
    get_model,
    get_model_registry,
    get_model_spec,
)

__all__ = [
    "CANCER_REGISTRY",
    "MODEL_REGISTRY",
    "ModelRegistry",
    "ModelSpec",
    "get_all_models",
    "get_cancer_capabilities",
    "get_cancer_capability",
    "get_models_for_cancer",
    "get_model",
    "get_model_registry",
    "get_model_spec",
]
