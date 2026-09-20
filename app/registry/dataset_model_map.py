"""
Dataset to model capability mapping layer.
"""

from __future__ import annotations


DATASET_MODEL_MAP = {

    "LUNA16": [
        "luna16_lung_segmentation",
        "luna16_nodule_detector",
    ],

    "BUSI": [
        "busi_breast_segmentation",
        "busi_breast_classifier",
    ],

}


def get_models_for_dataset(dataset_name: str):

    return DATASET_MODEL_MAP.get(
        dataset_name,
        []
    )


def get_dataset_for_model(model_id: str):

    for dataset, models in DATASET_MODEL_MAP.items():

        if model_id in models:
            return dataset

    return None