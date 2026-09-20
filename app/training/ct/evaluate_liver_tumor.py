"""
IRCADb01 Liver Tumor Segmentation Evaluation

Evaluates the trained 3D U-Net checkpoint on the same
deterministic validation split used during training.

Metrics:
- Liver Dice
- Tumor Dice
- Tumor precision
- Tumor recall
- Per-case results
- Aggregate results

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from app.models.ct.liver_tumor_unet import LiverTumor3DUNet
from app.training.ct.train_liver_tumor import LiverTumorROIDataset as PreprocessedLiverDataset


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT_PATH = (
    _PROJECT_ROOT
    / "checkpoints"
    / "liver_tumor"
    / "liver_tumor_best.pt"
)


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    class_id: int,
    epsilon: float = 1e-7,
):

    pred = (
        prediction == class_id
    )

    truth = (
        target == class_id
    )

    pred_count = int(
        pred.sum()
    )

    truth_count = int(
        truth.sum()
    )

    if (
        pred_count == 0
        and truth_count == 0
    ):
        return 1.0

    intersection = int(
        np.logical_and(
            pred,
            truth,
        ).sum()
    )

    return float(
        (
            2.0 * intersection
            +
            epsilon
        )
        /
        (
            pred_count
            +
            truth_count
            +
            epsilon
        )
    )


def precision_score(
    prediction: np.ndarray,
    target: np.ndarray,
    class_id: int,
    epsilon: float = 1e-7,
):

    pred = (
        prediction == class_id
    )

    truth = (
        target == class_id
    )

    true_positive = int(
        np.logical_and(
            pred,
            truth,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            pred,
            np.logical_not(truth),
        ).sum()
    )

    if (
        true_positive == 0
        and false_positive == 0
    ):
        return 0.0

    return float(
        true_positive
        /
        (
            true_positive
            +
            false_positive
            +
            epsilon
        )
    )


def recall_score(
    prediction: np.ndarray,
    target: np.ndarray,
    class_id: int,
    epsilon: float = 1e-7,
):

    pred = (
        prediction == class_id
    )

    truth = (
        target == class_id
    )

    true_positive = int(
        np.logical_and(
            pred,
            truth,
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            np.logical_not(pred),
            truth,
        ).sum()
    )

    if (
        true_positive == 0
        and false_negative == 0
    ):
        return 0.0

    return float(
        true_positive
        /
        (
            true_positive
            +
            false_negative
            +
            epsilon
        )
    )


def evaluate_liver_tumor(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    dataset = PreprocessedLiverDataset()

    train_size = max(
        1,
        int(
            len(dataset) * 0.8
        ),
    )

    val_size = (
        len(dataset)
        -
        train_size
    )

    if val_size == 0:
        train_size -= 1
        val_size = 1

    _, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size,
        ],
        generator=(
            torch.Generator()
            .manual_seed(42)
        ),
    )

    model = LiverTumor3DUNet(
        in_channels=1,
        out_channels=3,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    results = []

    liver_dice_values = []
    tumor_dice_values = []

    tumor_positive_dice_values = []

    tumor_precision_values = []
    tumor_recall_values = []

    print(
        f"Validation cases: {len(val_dataset)}"
    )

    print()

    with torch.no_grad():

        for index in range(
            len(val_dataset)
        ):

            sample = val_dataset[
                index
            ]

            image = sample[
                "image"
            ].unsqueeze(
                0
            ).to(
                device
            )

            target = sample[
                "label"
            ].cpu().numpy()

            case_name = sample[
                "case"
            ]

            use_amp = (
                device.type == "cuda"
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):

                logits = model(
                    image
                )

            prediction = torch.argmax(
                logits,
                dim=1,
            )[0].cpu().numpy()

            liver_dice = dice_score(
                prediction,
                target,
                class_id=1,
            )

            tumor_dice = dice_score(
                prediction,
                target,
                class_id=2,
            )

            tumor_precision = precision_score(
                prediction,
                target,
                class_id=2,
            )

            tumor_recall = recall_score(
                prediction,
                target,
                class_id=2,
            )

            target_tumor_voxels = int(
                np.count_nonzero(
                    target == 2
                )
            )

            predicted_tumor_voxels = int(
                np.count_nonzero(
                    prediction == 2
                )
            )

            has_true_tumor = (
                target_tumor_voxels > 0
            )

            liver_dice_values.append(
                liver_dice
            )

            tumor_dice_values.append(
                tumor_dice
            )

            tumor_precision_values.append(
                tumor_precision
            )

            tumor_recall_values.append(
                tumor_recall
            )

            if has_true_tumor:

                tumor_positive_dice_values.append(
                    tumor_dice
                )

            result = {

                "case":
                    case_name,

                "liver_dice":
                    liver_dice,

                "tumor_dice":
                    tumor_dice,

                "tumor_precision":
                    tumor_precision,

                "tumor_recall":
                    tumor_recall,

                "true_tumor_voxels":
                    target_tumor_voxels,

                "predicted_tumor_voxels":
                    predicted_tumor_voxels,
            }

            results.append(
                result
            )

            print(
                f"{case_name}"
            )

            print(
                f"  Liver Dice:       {liver_dice:.4f}"
            )

            print(
                f"  Tumor Dice:       {tumor_dice:.4f}"
            )

            print(
                f"  Tumor Precision:  {tumor_precision:.4f}"
            )

            print(
                f"  Tumor Recall:     {tumor_recall:.4f}"
            )

            print(
                f"  True tumor voxels: {target_tumor_voxels}"
            )

            print(
                f"  Pred tumor voxels: {predicted_tumor_voxels}"
            )

            print()

            del image
            del logits

            if device.type == "cuda":
                torch.cuda.empty_cache()

    mean_liver_dice = float(
        np.mean(
            liver_dice_values
        )
    )

    mean_tumor_dice = float(
        np.mean(
            tumor_dice_values
        )
    )

    if tumor_positive_dice_values:

        mean_positive_tumor_dice = float(
            np.mean(
                tumor_positive_dice_values
            )
        )

    else:

        mean_positive_tumor_dice = 0.0

    mean_tumor_precision = float(
        np.mean(
            tumor_precision_values
        )
    )

    mean_tumor_recall = float(
        np.mean(
            tumor_recall_values
        )
    )

    summary = {

        "validation_cases":
            len(val_dataset),

        "mean_liver_dice":
            mean_liver_dice,

        "mean_tumor_dice_all_cases":
            mean_tumor_dice,

        "mean_tumor_dice_positive_cases":
            mean_positive_tumor_dice,

        "mean_tumor_precision":
            mean_tumor_precision,

        "mean_tumor_recall":
            mean_tumor_recall,

        "checkpoint":
            str(
                checkpoint_path
            ),
    }

    print(
        "=============================="
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        f"Validation cases: {summary['validation_cases']}"
    )

    print(
        f"Mean Liver Dice: {mean_liver_dice:.4f}"
    )

    print(
        f"Mean Tumor Dice (all): {mean_tumor_dice:.4f}"
    )

    print(
        "Mean Tumor Dice "
        f"(tumor-positive cases): "
        f"{mean_positive_tumor_dice:.4f}"
    )

    print(
        f"Mean Tumor Precision: {mean_tumor_precision:.4f}"
    )

    print(
        f"Mean Tumor Recall: {mean_tumor_recall:.4f}"
    )

    return {
        "summary":
            summary,

        "cases":
            results,
    }


if __name__ == "__main__":

    evaluate_liver_tumor()
