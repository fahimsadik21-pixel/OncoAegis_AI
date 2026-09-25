"""Lazy specialist factory.

The managed API must be able to serve authentication, chat history, document
review, and educational guidance without importing PyTorch or every imaging
service at startup.  A selected specialist is imported only when an analysis
request actually needs it.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SPECIALIST_TARGETS: dict[str, tuple[str, str]] = {
    "luna16_lung_segmentation": (
        "app.imaging.ct.luna_inference",
        "LUNA16ModelService",
    ),
    "luna16_nodule_detector": (
        "app.imaging.ct.luna_nodule_inference",
        "LUNANoduleModelService",
    ),
    "busi_breast_segmentation": (
        "app.imaging.ultrasound.busi_inference",
        "BUSIModelService",
    ),
    "busi_breast_classifier": (
        "app.imaging.ultrasound.busi_inference",
        "BUSIModelService",
    ),
    "msd_brain_tumor_segmentation": (
        "app.imaging.mri.brain_tumor_inference",
        "BrainTumorModelService",
    ),
    "msd_pancreas_segmentation": (
        "app.imaging.ct.pancreas_analysis_service",
        "PancreasAnalysisService",
    ),
    "msd_pancreas_tumor_segmentation": (
        "app.imaging.ct.pancreas_analysis_service",
        "PancreasAnalysisService",
    ),
    "msd_colon_tumor_segmentation": (
        "app.imaging.ct.colon_analysis_service",
        "ColonAnalysisService",
    ),
    "ircadb01_liver_tumor_segmentation": (
        "app.imaging.ct.liver_analysis_service",
        "LiverAnalysisService",
    ),
    "isic2016_skin_lesion_segmentation": (
        "app.imaging.dermatology.skin_analysis_service",
        "SkinAnalysisService",
    ),
    "cnmc2019_all_cell_classifier": (
        "app.imaging.hematology.cnmc_analysis_service",
        "CNMCAnalysisService",
    ),
    "flowcap_aml_patient_classifier": (
        "app.imaging.hematology.flowcap_analysis_service",
        "FlowCAPAnalysisService",
    ),
    "tn3k_thyroid_nodule_segmentation": (
        "app.imaging.ultrasound.tn3k_analysis_service",
        "TN3KAnalysisService",
    ),
}


def get_specialist_class(model_id: str) -> type[Any]:
    """Return a specialist class, importing only its selected module."""

    try:
        module_name, class_name = _SPECIALIST_TARGETS[model_id]
    except KeyError as exc:
        raise ValueError(
            f"No specialist factory available for {model_id}"
        ) from exc

    module = import_module(module_name)
    return getattr(module, class_name)


def create_specialist(model_id: str) -> Any:
    return get_specialist_class(model_id)()
