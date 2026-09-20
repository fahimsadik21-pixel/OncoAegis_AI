"""
Pancreas Tumor Training V2

Goals:
- reduce full-volume background dominance
- crop around pancreas/cancer foreground
- oversample cancer-containing cases
- use focal-style CE + Dice loss
- keep memory suitable for RTX 3050 6GB

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
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from app.data.ct.msd_pancreas_loader import MSDPancreasDataset
from app.models.ct.pancreas_tumor_unet import PancreasTumor3DUNet


SEED = 42

TARGET_SIZE = (
    96,
    96,
    96,
)

ROI_MARGIN = (
    12,
    32,
    32,
)

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

DEFAULT_EPOCHS = 5

CHECKPOINT_DIR = (
    Path("checkpoints")
    / "msd_pancreas_tumor_v2"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def seed_everything(
    seed: int = SEED,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_ct(
    volume: np.ndarray,
):

    volume = np.clip(
        volume.astype(np.float32),
        -125.0,
        275.0,
    )

    volume = (
        volume + 125.0
    ) / 400.0

    return volume.astype(
        np.float32
    )


def get_foreground_bounds(
    label: np.ndarray,
):

    coords = np.argwhere(
        label > 0
    )

    if coords.size == 0:

        return (
            (0, label.shape[0]),
            (0, label.shape[1]),
            (0, label.shape[2]),
        )

    minimum = coords.min(
        axis=0
    )

    maximum = (
        coords.max(
            axis=0
        )
        +
        1
    )

    bounds = []

    for axis in range(3):

        start = max(
            0,
            int(
                minimum[axis]
                -
                ROI_MARGIN[axis]
            ),
        )

        end = min(
            label.shape[axis],
            int(
                maximum[axis]
                +
                ROI_MARGIN[axis]
            ),
        )

        bounds.append(
            (
                start,
                end,
            )
        )

    return tuple(bounds)


def crop_volume(
    image: np.ndarray,
    label: np.ndarray,
):

    bounds = get_foreground_bounds(
        label
    )

    (
        (z0, z1),
        (y0, y1),
        (x0, x1),
    ) = bounds

    return (
        image[
            z0:z1,
            y0:y1,
            x0:x1
        ],
        label[
            z0:z1,
            y0:y1,
            x0:x1
        ],
        bounds,
    )


def resize_image(
    image: np.ndarray,
):

    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                image
            )
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=TARGET_SIZE,
        mode="trilinear",
        align_corners=False,
    )

    return (
        tensor
        .squeeze(0)
        .squeeze(0)
    )


def resize_label(
    label: np.ndarray,
):

    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                label
            )
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=TARGET_SIZE,
        mode="nearest",
    )

    return (
        tensor
        .squeeze(0)
        .squeeze(0)
        .long()
    )


class PancreasROICancerDataset(
    Dataset
):

    def __init__(
        self,
    ):

        self.dataset = (
            MSDPancreasDataset()
        )

        self.indices = list(
            range(
                len(
                    self.dataset
                )
            )
        )

        self.cancer_indices = []

        print(
            "Scanning cancer-containing cases..."
        )

        for index in self.indices:

            sample = self.dataset[
                index
            ]

            if torch.any(
                sample["label"] == 2
            ):

                self.cancer_indices.append(
                    index
                )

        print(
            "Cancer-containing cases:",
            len(
                self.cancer_indices
            )
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

        image = normalize_ct(
            image
        )

        (
            image,
            label,
            bounds,
        ) = crop_volume(
            image,
            label,
        )

        image = resize_image(
            image
        )

        label = resize_label(
            label
        )

        image = image.unsqueeze(
            0
        )

        return {

            "image":
                image,

            "label":
                label,

            "case":
                sample["case"],

            "contains_cancer":
                bool(
                    torch.any(
                        label == 2
                    )
                ),

            "crop_bounds":
                bounds,
        }


class FocalCrossEntropyLoss(
    nn.Module
):

    def __init__(
        self,
        gamma: float = 2.0,
    ):

        super().__init__()

        self.gamma = gamma

        self.register_buffer(
            "class_weights",
            torch.tensor(
                [
                    0.1,
                    2.0,
                    15.0,
                ],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        logits,
        target,
    ):

        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
        )

        pt = torch.exp(
            -ce
        )

        focal = (
            (
                1.0
                -
                pt
            )
            **
            self.gamma
        ) * ce

        return focal.mean()


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
            F.one_hot(
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

        losses = []

        class_weights = {
            1: 1.0,
            2: 3.0,
        }

        for class_index in (
            1,
            2,
        ):

            pred = probabilities[
                :,
                class_index,
            ]

            truth = target_one_hot[
                :,
                class_index,
            ]

            intersection = (
                pred
                *
                truth
            ).sum()

            denominator = (
                pred.sum()
                +
                truth.sum()
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

            losses.append(
                class_weights[
                    class_index
                ]
                *
                (
                    1.0
                    -
                    dice
                )
            )

        return (
            sum(
                losses
            )
            /
            sum(
                class_weights.values()
            )
        )


class CombinedLoss(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.focal = (
            FocalCrossEntropyLoss()
        )

        self.dice = (
            ForegroundDiceLoss()
        )

    def forward(
        self,
        logits,
        target,
    ):

        focal = self.focal(
            logits,
            target,
        )

        dice = self.dice(
            logits,
            target,
        )

        return (
            0.4
            *
            focal
            +
            0.6
            *
            dice
        )


def binary_dice(
    prediction,
    target,
    class_index,
    smooth=1e-5,
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
        return None

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

    total_loss = 0.0

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

        total_loss += (
            loss.item()
            *
            images.size(0)
        )

    return (
        total_loss
        /
        len(
            loader.dataset
        )
    )


@torch.no_grad()
def validate(
    model,
    loader,
    loss_function,
):

    model.eval()

    total_loss = 0.0

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

        total_loss += (
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

            pancreas_dice = binary_dice(
                prediction[
                    batch_index
                ],
                labels[
                    batch_index
                ],
                1,
            )

            cancer_dice = binary_dice(
                prediction[
                    batch_index
                ],
                labels[
                    batch_index
                ],
                2,
            )

            if pancreas_dice is not None:
                pancreas_scores.append(
                    pancreas_dice
                )

            if cancer_dice is not None:
                cancer_scores.append(
                    cancer_dice
                )

    return {

        "loss":
            total_loss
            /
            len(
                loader.dataset
            ),

        "pancreas_dice":
            float(
                np.mean(
                    pancreas_scores
                )
            )
            if pancreas_scores
            else 0.0,

        "cancer_dice":
            float(
                np.mean(
                    cancer_scores
                )
            )
            if cancer_scores
            else 0.0,

        "cancer_cases_evaluated":
            len(
                cancer_scores
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

            "target_size":
                TARGET_SIZE,

            "labels":
                {
                    0:
                        "background",

                    1:
                        "pancreas",

                    2:
                        "cancer",
                },

            "model_name":
                "MSD Task07 Pancreas Tumor V2",
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
        PancreasROICancerDataset()
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

    (
        training_dataset,
        validation_dataset,
    ) = random_split(
        dataset,
        [
            training_size,
            validation_size,
        ],
        generator=generator,
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

    training_loader = DataLoader(

        training_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS
            >
            0
        ),
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS
            >
            0
        ),
    )

    model = (
        PancreasTumor3DUNet()
        .to(
            DEVICE
        )
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-5,
    )

    loss_function = (
        CombinedLoss()
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

    best_score = -1.0

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

        print(
            "Cancer cases evaluated:",
            validation[
                "cancer_cases_evaluated"
            ],
        )

        score = (
            validation[
                "pancreas_dice"
            ]
            +
            2.0
            *
            validation[
                "cancer_dice"
            ]
        )

        save_checkpoint(
            CHECKPOINT_DIR
            /
            "pancreas_tumor_v2_last.pt",
            model,
            optimizer,
            epoch,
            validation,
        )

        if score > best_score:

            best_score = score

            save_checkpoint(
                CHECKPOINT_DIR
                /
                "pancreas_tumor_v2_best.pt",
                model,
                optimizer,
                epoch,
                validation,
            )

            print(
                "Saved best checkpoint:",
                CHECKPOINT_DIR
                /
                "pancreas_tumor_v2_best.pt",
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