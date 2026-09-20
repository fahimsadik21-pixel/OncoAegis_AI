"""
Evaluate postprocessed liver locator ROI.

Purpose:
- Test the real inference-time liver locator
- Verify that predicted ROI contains the liver/tumor region
- Measure whether Stage-2 tumor segmentation can safely use the ROI

Research use only.
"""

from __future__ import annotations

import numpy as np

from app.data.ct.ircadb01_liver_loader import IRCADLiverDataset
from app.imaging.ct.liver_dicom_io import load_liver_ct_dicom_series
from app.imaging.ct.liver_locator_inference import (
    LiverLocatorInferenceService,
)


THRESHOLD = 0.80


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-7,
):

    prediction = prediction > 0
    target = target > 0

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    denominator = (
        prediction.sum()
        +
        target.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        (
            2.0 * intersection
            +
            epsilon
        )
        /
        (
            denominator
            +
            epsilon
        )
    )


def recall_score(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-7,
):

    prediction = prediction > 0
    target = target > 0

    true_positive = np.logical_and(
        prediction,
        target,
    ).sum()

    target_count = target.sum()

    if target_count == 0:
        return 1.0

    return float(
        true_positive
        /
        (
            target_count
            +
            epsilon
        )
    )


def roi_mask_from_bounds(
    shape,
    bounds,
):

    mask = np.zeros(
        shape,
        dtype=np.uint8,
    )

    if bounds is None:
        return mask

    (
        (z_min, z_max),
        (y_min, y_max),
        (x_min, x_max),
    ) = bounds

    mask[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1,
    ] = 1

    return mask


def evaluate():

    dataset = IRCADLiverDataset()

    locator = LiverLocatorInferenceService(
        liver_threshold=THRESHOLD
    )

    liver_dices = []
    liver_recalls = []
    tumor_roi_coverages = []
    roi_fractions = []

    failed_cases = []

    print(
        f"Evaluating {len(dataset)} cases "
        f"with threshold {THRESHOLD}"
    )

    print()

    for index in range(
        len(dataset)
    ):

        case_path = dataset.cases[
            index
        ]

        sample = dataset[
            index
        ]

        ground_truth = (
            sample["label"]
            .numpy()
        )

        true_liver = (
            ground_truth > 0
        ).astype(
            np.uint8
        )

        true_tumor = (
            ground_truth == 2
        ).astype(
            np.uint8
        )

        dicom = load_liver_ct_dicom_series(
            case_path
            / "PATIENT_DICOM"
        )

        prediction = locator.predict(
            dicom.volume
        )

        predicted_liver = (
            prediction.liver_mask
        )

        liver_dice = dice_score(
            predicted_liver,
            true_liver,
        )

        liver_recall = recall_score(
            predicted_liver,
            true_liver,
        )

        roi_mask = roi_mask_from_bounds(
            ground_truth.shape,
            prediction.crop_bounds,
        )

        total_voxels = int(
            np.prod(
                ground_truth.shape
            )
        )

        roi_voxels = int(
            roi_mask.sum()
        )

        roi_fraction = (
            roi_voxels
            /
            total_voxels
        )

        true_tumor_voxels = int(
            true_tumor.sum()
        )

        if true_tumor_voxels > 0:

            tumor_inside_roi = int(
                np.logical_and(
                    true_tumor > 0,
                    roi_mask > 0,
                ).sum()
            )

            tumor_roi_coverage = (
                tumor_inside_roi
                /
                true_tumor_voxels
            )

            tumor_roi_coverages.append(
                tumor_roi_coverage
            )

        else:

            tumor_roi_coverage = None

        liver_dices.append(
            liver_dice
        )

        liver_recalls.append(
            liver_recall
        )

        roi_fractions.append(
            roi_fraction
        )

        if (
            prediction.crop_bounds is None
            or liver_recall < 0.50
            or (
                tumor_roi_coverage is not None
                and tumor_roi_coverage < 0.95
            )
        ):

            failed_cases.append(
                case_path.name
            )

        print(
            case_path.name
        )

        print(
            f"  Liver Dice:       "
            f"{liver_dice:.4f}"
        )

        print(
            f"  Liver Recall:     "
            f"{liver_recall:.4f}"
        )

        print(
            f"  ROI fraction:     "
            f"{roi_fraction:.4f}"
        )

        print(
            f"  ROI bounds:       "
            f"{prediction.crop_bounds}"
        )

        if tumor_roi_coverage is None:

            print(
                "  Tumor ROI coverage: "
                "N/A (no tumor label)"
            )

        else:

            print(
                f"  Tumor ROI coverage: "
                f"{tumor_roi_coverage:.4f}"
            )

        print()

    print(
        "================================"
    )

    print(
        "LIVER LOCATOR ROI SUMMARY"
    )

    print(
        "================================"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Mean Liver Dice: "
        f"{np.mean(liver_dices):.4f}"
    )

    print(
        f"Mean Liver Recall: "
        f"{np.mean(liver_recalls):.4f}"
    )

    print(
        f"Mean ROI fraction: "
        f"{np.mean(roi_fractions):.4f}"
    )

    if tumor_roi_coverages:

        print(
            f"Mean Tumor ROI Coverage: "
            f"{np.mean(tumor_roi_coverages):.4f}"
        )

        print(
            f"Minimum Tumor ROI Coverage: "
            f"{np.min(tumor_roi_coverages):.4f}"
        )

    print(
        f"Problem cases: "
        f"{failed_cases}"
    )


if __name__ == "__main__":
    evaluate()