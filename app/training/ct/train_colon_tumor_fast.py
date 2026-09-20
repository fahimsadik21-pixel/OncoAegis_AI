"""
Fast cached trainer for MSD Task10 Colon tumor segmentation.

Cache:
    datasets/processed/msd_colon_96

Input:
    [1, 96, 96, 96]

Classes:
    0 = background
    1 = colon tumor

Research use only.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from app.models.ct.colon_tumor_unet import (
    ColonTumorUNet3D,
)


CACHE_DIR = Path(
    "datasets/processed/msd_colon_96"
)

CHECKPOINT_DIR = Path(
    "checkpoints/msd_colon_tumor"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    /
    "colon_tumor_best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    /
    "colon_tumor_last.pt"
)

SEED = 42


class CachedColonDataset(Dataset):

    def __init__(
        self,
        cache_dir: str | Path = CACHE_DIR,
    ):

        self.cache_dir = Path(
            cache_dir
        )

        self.files = sorted(
            self.cache_dir.glob(
                "*.pt"
            )
        )

        if not self.files:
            raise RuntimeError(
                f"No cache files found in {self.cache_dir}"
            )

        print(
            f"Preloading {len(self.files)} cached cases..."
        )

        self.samples = []

        for path in self.files:

            item = torch.load(
                path,
                map_location="cpu",
            )

            self.samples.append(
                {
                    "image":
                        item["image"].float(),

                    "label":
                        item["label"].long(),

                    "case":
                        item["case"],
                }
            )

        print(
            "Cache preload complete."
        )

    def __len__(
        self,
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        return self.samples[
            index
        ]


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
):

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    tumor_probability = probabilities[
        :,
        1,
    ]

    target_tumor = (
        target == 1
    ).float()

    intersection = torch.sum(
        tumor_probability
        *
        target_tumor
    )

    denominator = (
        torch.sum(
            tumor_probability
        )
        +
        torch.sum(
            target_tumor
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


def compute_tumor_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
):

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    prediction = (
        prediction == 1
    )

    target = (
        target == 1
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

    return float(
        dice.item()
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    training: bool,
):

    if training:
        model.train()
    else:
        model.eval()

    class_weights = torch.tensor(
        [
            0.15,
            8.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    ce_loss_function = nn.CrossEntropyLoss(
        weight=class_weights
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

            labels = batch[
                "label"
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

                ce = ce_loss_function(
                    logits,
                    labels,
                )

                dice = dice_loss(
                    logits,
                    labels,
                )

                loss = (
                    0.4
                    *
                    ce
                    +
                    0.6
                    *
                    dice
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

            total_dice += compute_tumor_dice(
                logits.detach(),
                labels,
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
        default=2,
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

    dataset = CachedColonDataset()

    validation_size = max(
        1,
        int(
            len(dataset)
            *
            0.20
        ),
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
        num_workers=0,
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
        num_workers=0,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
    )

    model = ColonTumorUNet3D().to(
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

        validation_loss, validation_dice = run_epoch(
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
            f"Val Loss {validation_loss:.4f} | "
            f"Val Tumor Dice {validation_dice:.4f}"
        )

        checkpoint = {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_dice":
                max(
                    best_dice,
                    validation_dice,
                ),

            "validation_dice":
                validation_dice,

            "model_name":
                "MSD Task10 Colon Tumor 3D U-Net",

            "model_version":
                f"research-baseline-epoch-{epoch}",

            "input_size":
                (
                    96,
                    96,
                    96,
                ),

            "labels": {
                0:
                    "background",

                1:
                    "colon cancer primary",
            },
        }

        torch.save(
            checkpoint,
            LAST_CHECKPOINT,
        )

        if validation_dice > best_dice:

            best_dice = validation_dice

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
        "Best validation tumor Dice:",
        best_dice
    )


if __name__ == "__main__":
    main()