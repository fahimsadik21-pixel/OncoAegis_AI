"""
Train MSD Task07 Pancreas CT segmentation model.

Classes:
    0 = background
    1 = pancreas
    2 = cancer

Research baseline only.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

from app.data.ct.msd_pancreas_loader import MSDPancreasDataset
from app.imaging.ct.pancreas_preprocessing import preprocess_pancreas_volume
from app.models.ct.pancreas_tumor_unet import PancreasTumor3DUNet


SEED = 42

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEFAULT_EPOCHS = 10

CHECKPOINT_DIR = Path(
    "checkpoints"
) / "msd_pancreas_tumor"


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


def seed_everything(
    seed: int = SEED,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


class PancreasTrainingDataset(Dataset):

    def __init__(
        self,
    ):

        self.dataset = (
            MSDPancreasDataset()
        )

    def __len__(
        self,
    ):

        return len(
            self.dataset
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.dataset[
            index
        ]

        image = (
            sample["image"]
            .squeeze(0)
            .numpy()
        )

        label = (
            sample["label"]
            .numpy()
        )

        processed = (
            preprocess_pancreas_volume(
                image,
                label,
            )
        )

        return {

            "image":
                processed.image,

            "label":
                processed.label,

            "case":
                sample["case"],
        }


class ForegroundDiceLoss(
    nn.Module
):

    def __init__(
        self,
        smooth: float = 1e-5,
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        target,
    ):

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        target_one_hot = (
            torch.nn.functional.one_hot(
                target,
                num_classes=3,
            )
            .permute(
                0,
                4,
                1,
                2,
                3,
            )
            .float()
        )

        dice_losses = []

        # Ignore background class 0
        for class_index in (
            1,
            2,
        ):

            prediction_class = (
                probabilities[
                    :,
                    class_index,
                ]
            )

            target_class = (
                target_one_hot[
                    :,
                    class_index,
                ]
            )

            intersection = (
                prediction_class
                *
                target_class
            ).sum()

            denominator = (
                prediction_class.sum()
                +
                target_class.sum()
            )

            dice = (
                2.0
                *
                intersection
                +
                self.smooth
            ) / (
                denominator
                +
                self.smooth
            )

            dice_losses.append(
                1.0 - dice
            )

        return torch.stack(
            dice_losses
        ).mean()


class PancreasWeightedLoss(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.cross_entropy = (
            nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [
                        0.25,
                        2.0,
                        8.0,
                    ],
                    dtype=torch.float32,
                )
            )
        )

        self.dice = (
            ForegroundDiceLoss()
        )

    def forward(
        self,
        logits,
        target,
    ):

        ce = self.cross_entropy(
            logits,
            target,
        )

        dice = self.dice(
            logits,
            target,
        )

        return (
            0.5
            *
            ce
            +
            0.5
            *
            dice
        )


def binary_dice(
    prediction,
    target,
    class_index: int,
    smooth: float = 1e-5,
):

    prediction_mask = (
        prediction
        ==
        class_index
    )

    target_mask = (
        target
        ==
        class_index
    )

    prediction_sum = (
        prediction_mask.sum().item()
    )

    target_sum = (
        target_mask.sum().item()
    )

    if (
        prediction_sum == 0
        and
        target_sum == 0
    ):

        return 1.0

    intersection = (
        prediction_mask
        &
        target_mask
    ).sum().item()

    return (
        2.0
        *
        intersection
        +
        smooth
    ) / (
        prediction_sum
        +
        target_sum
        +
        smooth
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    scaler,
):

    model.train()

    running_loss = 0.0

    for batch in loader:

        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type
                ==
                "cuda"
            ),
        ):

            logits = model(
                images
            )

            loss = loss_function(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

    return (
        running_loss
        /
        len(loader.dataset)
    )


@torch.no_grad()
def validate(
    model,
    loader,
    loss_function,
):

    model.eval()

    running_loss = 0.0

    pancreas_scores = []

    cancer_scores = []

    for batch in loader:

        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type
                ==
                "cuda"
            ),
        ):

            logits = model(
                images
            )

            loss = loss_function(
                logits,
                labels,
            )

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        prediction = torch.argmax(
            logits,
            dim=1,
        )

        for batch_index in range(
            prediction.shape[0]
        ):

            pancreas_scores.append(
                binary_dice(
                    prediction[
                        batch_index
                    ],
                    labels[
                        batch_index
                    ],
                    class_index=1,
                )
            )

            cancer_scores.append(
                binary_dice(
                    prediction[
                        batch_index
                    ],
                    labels[
                        batch_index
                    ],
                    class_index=2,
                )
            )

    return {

        "loss":
            running_loss
            /
            len(loader.dataset),

        "pancreas_dice":
            float(
                np.mean(
                    pancreas_scores
                )
            ),

        "cancer_dice":
            float(
                np.mean(
                    cancer_scores
                )
            ),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation":
                validation,

            "model_name":
                "MSD Task07 Pancreas 3D U-Net",

            "labels":
                {
                    0:
                        "background",

                    1:
                        "pancreas",

                    2:
                        "cancer",
                },
        },
        path,
    )


def train(
    epochs: int = DEFAULT_EPOCHS,
):

    seed_everything()

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    dataset = (
        PancreasTrainingDataset()
    )

    total_cases = len(
        dataset
    )

    validation_size = max(
        1,
        int(
            total_cases
            *
            0.2
        ),
    )

    training_size = (
        total_cases
        -
        validation_size
    )

    generator = (
        torch.Generator()
        .manual_seed(
            SEED
        )
    )

    training_dataset, validation_dataset = (
        random_split(
            dataset,
            [
                training_size,
                validation_size,
            ],
            generator=generator,
        )
    )

    print(
        "Training cases:",
        len(
            training_dataset
        )
    )

    print(
        "Validation cases:",
        len(
            validation_dataset
        )
    )

    training_loader = (
        DataLoader(
            training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(
                DEVICE.type
                ==
                "cuda"
            ),
        )
    )

    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(
                DEVICE.type
                ==
                "cuda"
            ),
        )
    )

    model = (
        PancreasTumor3DUNet()
        .to(
            DEVICE
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5,
        )
    )

    loss_function = (
        PancreasWeightedLoss()
        .to(
            DEVICE
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type
            ==
            "cuda"
        ),
    )

    best_validation_loss = (
        float(
            "inf"
        )
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            loss_function,
            scaler,
        )

        validation = validate(
            model,
            validation_loader,
            loss_function,
        )

        print(
            f"Epoch {epoch}/{epochs}"
        )

        print(
            f"Train Loss: "
            f"{training_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{validation['loss']:.4f}"
        )

        print(
            f"Pancreas Dice: "
            f"{validation['pancreas_dice']:.4f}"
        )

        print(
            f"Cancer Dice: "
            f"{validation['cancer_dice']:.4f}"
        )

        last_checkpoint = (
            CHECKPOINT_DIR
            /
            "pancreas_tumor_last.pt"
        )

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch,
            validation,
        )

        if (
            validation[
                "loss"
            ]
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation[
                    "loss"
                ]
            )

            best_checkpoint = (
                CHECKPOINT_DIR
                /
                "pancreas_tumor_best.pt"
            )

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                validation,
            )

            print(
                "Saved best checkpoint:",
                best_checkpoint
            )

        print(
            "-" * 60
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs
    )