"""High-level capability view kept compatible with the original API."""

from __future__ import annotations

from copy import deepcopy

from app.registry.cancer_registry import get_cancer_capabilities


CANCER_CAPABILITIES = {

    "lung_cancer": {
        "organs": ["lung"],
        "modalities": ["CT", "DOCUMENT"],
        "detection": True,
        "segmentation": True,
        "classification": True,
        "status": "planned"
    },

    "liver_cancer": {
        "organs": ["liver"],
        "modalities": ["CT", "ULTRASOUND", "DOCUMENT"],
        "detection": True,
        "segmentation": True,
        "classification": True,
        "status": "planned"
    },

    "kidney_cancer": {
        "organs": ["kidney"],
        "modalities": ["CT", "ULTRASOUND", "DOCUMENT"],
        "detection": True,
        "segmentation": True,
        "classification": True,
        "status": "planned"
    },

    "pancreatic_cancer": {
        "organs": ["pancreas"],
        "modalities": ["CT", "DOCUMENT"],
        "detection": True,
        "segmentation": True,
        "classification": True,
        "status": "planned"
    },

    "breast_cancer": {
        "organs": ["breast"],
        "modalities": ["ULTRASOUND", "DOCUMENT"],
        "detection": True,
        "segmentation": True,
        "classification": True,
        "status": "research_baseline",
        "available_pipeline": "BUSI ultrasound segmentation + classification",
        "clinical_validation": False
    },

   "thyroid_cancer": {
    "organs": ["thyroid"],
    "modalities": ["ULTRASOUND", "DOCUMENT"],
    "detection": False,
    "segmentation": True,
    "classification": False,
    "status": "research_baseline",
    "available_pipeline": (
        "TN3K thyroid nodule ultrasound segmentation"
    ),
    "clinical_validation": False,
   },

    "prostate_cancer": {
        "organs": ["prostate"],
        "modalities": ["DOCUMENT"],
        "detection": False,
        "segmentation": False,
        "classification": True,
        "status": "planned"
    },

      "leukemia": {
        "organs": ["blood"],
        "modalities": [
            "FLOW_CYTOMETRY",
            "MICROSCOPY",
            "DOCUMENT",
            "TEXT",
        ],
        "detection": False,
        "segmentation": False,
        "classification": True,
        "status": "research_baseline",
        "available_pipelines": [
            "DREAM6 FlowCAP-II patient-level AML classifier",
            "C-NMC 2019 single-cell ALL classifier",
        ],
        "clinical_validation": False,
    },
 }


def get_capabilities():
    """Return high-level capabilities enriched with registered model IDs."""

    capabilities = deepcopy(CANCER_CAPABILITIES)
    registered = get_cancer_capabilities()
    for cancer_name, entry in registered.items():
        if cancer_name not in capabilities:
            continue
        capabilities[cancer_name]["registered_models"] = list(entry["models"])
        capabilities[cancer_name]["registry_status"] = entry["status"]
        capabilities[cancer_name]["clinical_validation"] = entry[
            "clinical_validation"
        ]
    return capabilities
