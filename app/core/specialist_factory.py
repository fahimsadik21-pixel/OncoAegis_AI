from __future__ import annotations


from app.imaging.ct.luna_inference import (
    LUNA16ModelService,
)

from app.imaging.ct.luna_nodule_inference import (
    LUNANoduleModelService,
)

from app.imaging.ultrasound.busi_inference import (
    BUSIModelService,
)

from app.imaging.mri.brain_tumor_inference import (
    BrainTumorModelService,
)

from app.imaging.ct.pancreas_analysis_service import (
    PancreasAnalysisService,
)

from app.imaging.ct.colon_analysis_service import (
    ColonAnalysisService,
)

from app.imaging.ct.liver_analysis_service import (
    LiverAnalysisService,
)

from app.imaging.dermatology.skin_analysis_service import (
    SkinAnalysisService,
)

from app.imaging.hematology.cnmc_analysis_service import (
    CNMCAnalysisService,
)

from app.imaging.hematology.flowcap_analysis_service import (
    FlowCAPAnalysisService,
)

from app.imaging.ultrasound.tn3k_analysis_service import (
    TN3KAnalysisService,
)


_SPECIALIST_CLASSES = {
    "luna16_lung_segmentation": LUNA16ModelService,
    "luna16_nodule_detector": LUNANoduleModelService,
    "busi_breast_segmentation": BUSIModelService,
    "busi_breast_classifier": BUSIModelService,
    "msd_brain_tumor_segmentation": BrainTumorModelService,
    "msd_pancreas_segmentation": PancreasAnalysisService,
    "msd_pancreas_tumor_segmentation": PancreasAnalysisService,
    "msd_colon_tumor_segmentation": ColonAnalysisService,
    "ircadb01_liver_tumor_segmentation": LiverAnalysisService,
    "isic2016_skin_lesion_segmentation": SkinAnalysisService,
    "cnmc2019_all_cell_classifier": CNMCAnalysisService,
    "flowcap_aml_patient_classifier": FlowCAPAnalysisService,
    "tn3k_thyroid_nodule_segmentation": TN3KAnalysisService,
}


def get_specialist_class(model_id: str):
    """Return the service class without constructing or loading a model."""

    try:
        return _SPECIALIST_CLASSES[model_id]
    except KeyError as exc:
        raise ValueError(
            f"No specialist factory available for {model_id}"
        ) from exc


def create_specialist(model_id: str):
    return get_specialist_class(model_id)()
