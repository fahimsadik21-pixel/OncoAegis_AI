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



def create_specialist(
    model_id: str
):

    # -----------------
    # CT Lung
    # -----------------

    if model_id == "luna16_lung_segmentation":
        return LUNA16ModelService()

    if model_id == "luna16_nodule_detector":
        return LUNANoduleModelService()



    # -----------------
    # Breast Ultrasound
    # -----------------

    if model_id in {
        "busi_breast_segmentation",
        "busi_breast_classifier",
    }:
        return BUSIModelService()



    # -----------------
    # Brain MRI
    # -----------------

    if model_id == "msd_brain_tumor_segmentation":
        return BrainTumorModelService()



    # -----------------
    # Pancreas CT
    # -----------------

    if model_id in {
        "msd_pancreas_segmentation",
        "msd_pancreas_tumor_segmentation",
    }:
        return PancreasAnalysisService()



    # -----------------
    # Colon CT
    # -----------------

    if model_id == "msd_colon_tumor_segmentation":
        return ColonAnalysisService()



    # -----------------
    # Liver CT
    # -----------------

    if model_id == "ircadb01_liver_tumor_segmentation":
        return LiverAnalysisService()



    # -----------------
    # Skin Dermoscopy
    # -----------------

    if model_id == "isic2016_skin_lesion_segmentation":
        return SkinAnalysisService()



    # -----------------
    # Blood microscopy
    # -----------------

    if model_id == "cnmc2019_all_cell_classifier":
        return CNMCAnalysisService()



    # -----------------
    # Blood flow cytometry
    # -----------------

    if model_id == "flowcap_aml_patient_classifier":
        return FlowCAPAnalysisService()



    # -----------------
    # Thyroid ultrasound
    # -----------------

    if model_id == "tn3k_thyroid_nodule_segmentation":
        return TN3KAnalysisService()



    raise ValueError(
        f"No specialist factory available for {model_id}"
    )
