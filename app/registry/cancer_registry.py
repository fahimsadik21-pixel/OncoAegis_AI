"""High-level cancer capability registry.

The registry maps clinical areas to research specialist model IDs. A listed
model is not a clinical diagnosis and does not imply clinical validation.
"""

from __future__ import annotations

from typing import Any

from app.registry.model_registry import get_model_registry, normalize_modality


_RESEARCH_SAFETY = (
    "Registered outputs are research evidence only, require expert review, "
    "and cannot confirm or exclude malignancy."
)


CANCER_REGISTRY: dict[str, dict[str, Any]] = {
    "lung_cancer": {
        "organs": ["lung"],
        "modalities": ["CT"],
        "models": [
            "luna16_lung_segmentation",
            "luna16_nodule_detector",
        ],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": (
            "LUNA16 segments lung anatomy and identifies annotated-nodule-like "
            "patches; it does not provide pathology-confirmed cancer labels."
        ),
    },
    "breast_cancer": {
        "organs": ["breast"],
        "modalities": ["ULTRASOUND"],
        "models": [
            "busi_breast_segmentation",
            "busi_breast_classifier",
        ],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": (
            "BUSI classes are dataset labels and must not be presented as a "
            "confirmed cancer diagnosis."
        ),
    },
    "brain_cancer": {
        "organs": ["brain"],
        "modalities": ["MRI"],
        "models": ["msd_brain_tumor_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": _RESEARCH_SAFETY,
    },
    "liver_cancer": {
        "organs": ["liver"],
        "modalities": ["CT"],
        "models": ["ircadb01_liver_tumor_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": (
            "IRCADb01 liver and tumor masks are research segmentation output; "
            "they cannot confirm liver cancer."
        ),
    },
    "pancreatic_cancer": {
        "organs": ["pancreas"],
        "modalities": ["CT"],
        "models": ["msd_pancreas_tumor_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": _RESEARCH_SAFETY,
    },
    "colon_cancer": {
        "organs": ["colon"],
        "modalities": ["CT"],
        "models": ["msd_colon_tumor_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": _RESEARCH_SAFETY,
    },
    "colorectal_cancer": {
        "organs": ["colon"],
        "modalities": ["CT"],
        "models": ["msd_colon_tumor_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": _RESEARCH_SAFETY,
    },
    "skin_cancer": {
        "organs": ["skin"],
        "modalities": ["DERMOSCOPY"],
        "models": ["isic2016_skin_lesion_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": (
            "ISIC2016 lesion segmentation does not classify melanoma or "
            "confirm skin cancer."
        ),
    },
    "blood_cancer": {
        "organs": ["blood"],
        "modalities": ["MICROSCOPY", "FLOW_CYTOMETRY"],
        "models": [
            "cnmc2019_all_cell_classifier",
            "flowcap_aml_patient_classifier",
        ],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": (
            "C-NMC and FlowCAP outputs are research classifications or scores; "
            "hematopathology review and clinical testing are required."
        ),
    },
    "acute_lymphoblastic_leukemia": {
        "organs": ["blood"],
        "modalities": ["MICROSCOPY"],
        "models": ["cnmc2019_all_cell_classifier"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": _RESEARCH_SAFETY,
    },
    "acute_myeloid_leukemia": {
        "organs": ["blood"],
        "modalities": ["FLOW_CYTOMETRY"],
        "models": ["flowcap_aml_patient_classifier"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": _RESEARCH_SAFETY,
    },
    "thyroid_cancer": {
        "organs": ["thyroid"],
        "modalities": ["ULTRASOUND"],
        "models": ["tn3k_thyroid_nodule_segmentation"],
        "status": "research_baseline",
        "clinical_validation": False,
        "safety_note": (
            "TN3K segments thyroid nodule regions; it does not classify or "
            "confirm thyroid malignancy."
        ),
    },
}


def _validate_references() -> None:
    model_registry = get_model_registry()
    for cancer_name, capability in CANCER_REGISTRY.items():
        for model_id in capability["models"]:
            if model_registry.get_spec(model_id) is None:
                raise ValueError(
                    f"{cancer_name} references unknown model: {model_id}"
                )
        capability["modalities"] = [
            normalize_modality(modality)
            for modality in capability["modalities"]
        ]


_validate_references()


def get_cancer_capabilities() -> dict[str, dict[str, Any]]:
    return CANCER_REGISTRY


def get_cancer_capability(cancer_name: str) -> dict[str, Any] | None:
    return CANCER_REGISTRY.get(cancer_name.strip().lower())


def get_models_for_cancer(
    cancer_name: str,
    *,
    available_only: bool = False,
) -> tuple[str, ...]:
    capability = get_cancer_capability(cancer_name)
    if capability is None:
        return ()
    registry = get_model_registry()
    model_ids = []
    for model_id in capability["models"]:
        spec = registry.get_spec(model_id)
        if spec is not None and (not available_only or spec.checkpoint_available):
            model_ids.append(model_id)
    return tuple(model_ids)
