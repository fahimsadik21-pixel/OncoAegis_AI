from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.data.ultrasound.tn3k_dataset import TN3KDataset
from app.models.ultrasound.tn3k_unet import TN3KUNet


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "tn3k_segmentation"
    / "tn3k_unet_best.pt"
)


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:

    prediction = prediction.flatten(1)
    target = target.flatten(1)

    intersection = (
        prediction * target
    ).sum(dim=1)

    denominator = (
        prediction.sum(dim=1)
        + target.sum(dim=1)
    )

    return (
        (2.0 * intersection + eps)
        / (denominator + eps)
    )


def iou_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:

    prediction = prediction.flatten(1)
    target = target.flatten(1)

    intersection = (
        prediction * target
    ).sum(dim=1)

    union = (
        prediction.sum(dim=1)
        + target.sum(dim=1)
        - intersection
    )

    return (
        (intersection + eps)
        / (union + eps)
    )


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DEVICE:", device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    image_size = int(
        checkpoint.get(
            "image_size",
            256,
        )
    )

    model = TN3KUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    dataset = TN3KDataset(
        "test",
        image_size=image_size,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            masks = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            dice = dice_score(
                predictions,
                masks,
            )

            iou = iou_score(
                predictions,
                masks,
            )

            batch_size = (
                images.size(0)
            )

            total_dice += (
                dice.sum().item()
            )

            total_iou += (
                iou.sum().item()
            )

            total_samples += (
                batch_size
            )

    print(
        "TEST SAMPLES:",
        total_samples,
    )

    print(
        "TEST DICE:",
        f"{total_dice / total_samples:.6f}",
    )

    print(
        "TEST IOU:",
        f"{total_iou / total_samples:.6f}",
    )


if __name__ == "__main__":
    main()