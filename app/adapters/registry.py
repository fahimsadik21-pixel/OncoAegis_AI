from __future__ import annotations

from typing import Any

from app.adapters.busi_adapter import BUSIAdapter
from app.adapters.luna_adapter import LUNAAdapter
from app.adapters.luna_nodule_adapter import LUNANoduleAdapter
from app.adapters.brain_mri_adapter import BrainMRIAdapter
from app.adapters.pancreas_ct_adapter import PancreasCTAdapter
from app.adapters.colon_ct_adapter import ColonCTAdapter
from app.adapters.liver_ct_adapter import LiverCTAdapter
from app.adapters.skin_adapter import SkinAdapter
from app.adapters.cnmc_adapter import CNMCAdapter
from app.adapters.flowcap_adapter import FlowCAPAdapter
from app.adapters.tn3k_adapter import TN3KAdapter


_ADAPTERS = {

    "busi_breast_segmentation":
        BUSIAdapter(),

    "busi_breast_classifier":
        BUSIAdapter(),


    "luna16_lung_segmentation":
        LUNAAdapter(),

    "luna16_nodule_detector":
        LUNANoduleAdapter(),


    "msd_brain_tumor_segmentation":
        BrainMRIAdapter(),


    "msd_pancreas_segmentation":
        PancreasCTAdapter(),

    "msd_pancreas_tumor_segmentation":
        PancreasCTAdapter(),


    "msd_colon_tumor_segmentation":
        ColonCTAdapter(),


    "ircadb01_liver_tumor_segmentation":
        LiverCTAdapter(),


    "isic2016_skin_lesion_segmentation":
        SkinAdapter(),


    "cnmc2019_all_cell_classifier":
        CNMCAdapter(),


    "flowcap_aml_patient_classifier":
        FlowCAPAdapter(),

    "tn3k_thyroid_nodule_segmentation":
        TN3KAdapter(),

}


def get_adapter(model_id: str):

    adapter = _ADAPTERS.get(model_id)

    if adapter is None:
        raise ValueError(
            f"No adapter registered for model: {model_id}"
        )

    return adapter



def list_adapters() -> dict[str, Any]:

    return {
        model_id: adapter.metadata()
        for model_id, adapter in _ADAPTERS.items()
    }
