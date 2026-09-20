"""
Fast ISIC2016 skin lesion segmentation trainer.

Research use only.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split,
)

from app.data.dermatology.isic2016_loader import (
    ISIC2016SegmentationDataset,
)

from app.imaging.dermatology.skin_preprocessing import (
    preprocess_skin_image,
    preprocess_skin_mask,
)

from app.models.dermatology.skin_lesion_unet import (
    SkinLesionUNet2D,
)


CHECKPOINT_DIR = Path(
    "checkpoints/isic2016_segmentation"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    /
    "isic2016_skin_segmentation_best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    /
    "isic2016_skin_segmentation_last.pt"
)

SEED = 42


class ProcessedSkinDataset(torch.utils.data.Dataset):

    def __init__(
        self,
    ):

        self.base = (
            ISIC2016SegmentationDataset()
        )

    def __len__(
        self,
    ):
        return len(
            self.base
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.base[
            index
        ]

        image = preprocess_skin_image(
            sample[
                "image"
            ]
        )

        mask = preprocess_skin_mask(
            sample[
                "mask"
            ]
        )

        return {

            "image":
                image,

            "mask":
                mask,

            "case":
                sample[
                    "case"
                ],
        }


def dice_loss(
    logits,
    target,
    smooth=1e-5,
):

    probability = torch.softmax(
        logits,
        dim=1,
    )[
        :,
        1,
    ]

    target_fg = (
        target
        ==
        1
    ).float()

    intersection = torch.sum(
        probability
        *
        target_fg
    )

    denominator = (
        torch.sum(
            probability
        )
        +
        torch.sum(
            target_fg
        )
    )

    dice = (
        2.0
        *
        intersection
        +
        smooth
    ) / (
        denominator
        +
        smooth
    )

    return (
        1.0
        -
        dice
    )


def compute_dice(
    logits,
    target,
    smooth=1e-5,
):

    prediction = (
        torch.argmax(
            logits,
            dim=1,
        )
        ==
        1
    )

    target = (
        target
        ==
        1
    )

    intersection = (
        prediction
        &
        target
    ).sum().float()

    denominator = (
        prediction.sum().float()
        +
        target.sum().float()
    )

    if denominator.item() == 0:
        return 1.0

    return float(
        (
            (
                2.0
                *
                intersection
                +
                smooth
            )
            /
            (
                denominator
                +
                smooth
            )
        ).item()
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [
                0.4,
                1.6,
            ],
            device=device,
        )
    )

    total_loss = 0.0
    total_dice = 0.0
    batches = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

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

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type
                    ==
                    "cuda"
                ),
            ):

                logits = model(
                    images
                )

                ce = ce_loss(
                    logits,
                    masks,
                )

                dl = dice_loss(
                    logits,
                    masks,
                )

                loss = (
                    0.4
                    *
                    ce
                    +
                    0.6
                    *
                    dl
                )

            if training:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            total_loss += float(
                loss.item()
            )

            total_dice += compute_dice(
                logits.detach(),
                masks,
            )

            batches += 1

    return (
        total_loss
        /
        max(
            batches,
            1,
        ),
        total_dice
        /
        max(
            batches,
            1,
        ),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    dataset = ProcessedSkinDataset()

    validation_size = int(
        len(dataset)
        *
        0.20
    )

    training_size = (
        len(dataset)
        -
        validation_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [
            training_size,
            validation_size,
        ],
        generator=generator,
    )

    print(
        "Train:",
        len(
            train_dataset
        )
    )

    print(
        "Validation:",
        len(
            validation_dataset
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    model = SkinLesionUNet2D().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            ==
            "cuda"
        ),
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_dice = -1.0
    start_epoch = 1

    if LAST_CHECKPOINT.exists():

        print(
            "Resuming:",
            LAST_CHECKPOINT
        )

        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            +
            1
        )

        best_dice = float(
            checkpoint.get(
                "best_dice",
                -1.0,
            )
        )

    for epoch in range(
        start_epoch,
        start_epoch
        +
        args.epochs,
    ):

        train_loss, train_dice = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            training=True,
        )

        val_loss, val_dice = run_epoch(
            model,
            validation_loader,
            optimizer,
            scaler,
            device,
            training=False,
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Dice {train_dice:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Dice {val_dice:.4f}"
        )

        checkpoint = {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_dice":
                val_dice,

            "best_dice":
                max(
                    best_dice,
                    val_dice,
                ),

            "model_name":
                "ISIC2016 Skin Lesion 2D U-Net",

            "model_version":
                f"research-baseline-epoch-{epoch}",

            "input_size":
                (
                    256,
                    256,
                ),
        }

        torch.save(
            checkpoint,
            LAST_CHECKPOINT,
        )

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                checkpoint,
                BEST_CHECKPOINT,
            )

            print(
                "Saved BEST checkpoint:",
                BEST_CHECKPOINT
            )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "Best validation Dice:",
        best_dice
    )


if __name__ == "__main__":
    main()