from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from app.models.ct.pancreas_tumor_unet import PancreasTumor3DUNet


SEED = 42

CACHE_DIR = Path(
    "datasets/processed/msd_pancreas_96"
)

CHECKPOINT_DIR = Path(
    "checkpoints/msd_pancreas_tumor_fast"
)

DEFAULT_RESUME = Path(
    "checkpoints/msd_pancreas_tumor_v2/"
    "pancreas_tumor_v2_best.pt"
)

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


torch.backends.cudnn.benchmark = True


def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_number(path: Path):

    try:
        return int(
            path.stem.split("_")[-1]
        )
    except Exception:
        return 0


class CachedPancreasDataset(Dataset):

    def __init__(
        self,
        preload: bool = True,
    ):

        if not CACHE_DIR.exists():
            raise RuntimeError(
                f"Cache directory not found: {CACHE_DIR}"
            )

        self.files = sorted(
            CACHE_DIR.glob("*.pt"),
            key=case_number,
        )

        if not self.files:
            raise RuntimeError(
                "No cached pancreas files found."
            )

        print(
            "Cached cases:",
            len(self.files)
        )

        self.preload = preload
        self.cached_data = None

        if preload:

            print(
                "Preloading cached dataset into RAM..."
            )

            self.cached_data = []

            for index, path in enumerate(
                self.files
            ):

                item = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )

                self.cached_data.append(
                    item
                )

                if (
                    (index + 1) % 50 == 0
                    or
                    index + 1 == len(self.files)
                ):

                    print(
                        f"Loaded "
                        f"{index + 1}/"
                        f"{len(self.files)}"
                    )

            print(
                "RAM preload complete."
            )

    def __len__(self):

        return len(
            self.files
        )

    def __getitem__(
        self,
        index,
    ):

        if self.cached_data is not None:

            item = self.cached_data[
                index
            ]

        else:

            item = torch.load(
                self.files[index],
                map_location="cpu",
                weights_only=False,
            )

        image = item[
            "image"
        ]

        label = item[
            "label"
        ]

        if image.ndim == 3:
            image = image.unsqueeze(0)

        return {
            "image":
                image,

            "label":
                label.long(),

            "case":
                item["case"],
        }


class FocalCrossEntropyLoss(
    nn.Module
):

    def __init__(
        self,
        gamma=2.0,
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
            (1.0 - pt)
            ** self.gamma
        ) * ce

        return focal.mean()


class ForegroundDiceLoss(
    nn.Module
):

    def __init__(
        self,
        smooth=1e-5,
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

        weights = {
            1: 1.0,
            2: 3.0,
        }

        for class_index in (
            1,
            2,
        ):

            prediction = probabilities[
                :,
                class_index,
            ]

            truth = target_one_hot[
                :,
                class_index,
            ]

            intersection = (
                prediction
                *
                truth
            ).sum()

            denominator = (
                prediction.sum()
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
                weights[class_index]
                *
                (
                    1.0
                    -
                    dice
                )
            )

        return (
            sum(losses)
            /
            sum(weights.values())
        )


class CombinedLoss(
    nn.Module
):

    def __init__(self):

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
            0.4 * focal
            +
            0.6 * dice
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
            dtype=torch.float32,
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
        len(loader.dataset)
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
            dtype=torch.float32,
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

        for i in range(
            prediction.shape[0]
        ):

            pancreas = binary_dice(
                prediction[i],
                labels[i],
                1,
            )

            cancer = binary_dice(
                prediction[i],
                labels[i],
                2,
            )

            if pancreas is not None:
                pancreas_scores.append(
                    pancreas
                )

            if cancer is not None:
                cancer_scores.append(
                    cancer
                )

    return {
        "loss":
            total_loss
            /
            len(loader.dataset),

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
                (
                    96,
                    96,
                    96,
                ),

            "model_name":
                "MSD Task07 Pancreas Fast Cached V2",

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
    epochs=3,
    batch_size=BATCH_SIZE,
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

    dataset = CachedPancreasDataset(
        preload=True
    )

    validation_size = int(
        len(dataset)
        *
        0.2
    )

    training_size = (
        len(dataset)
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

    print(
        "Batch size:",
        batch_size
    )

    training_loader = DataLoader(

        training_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            ==
            "cuda"
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

    start_epoch = 1

    if DEFAULT_RESUME.exists():

        print(
            "Resuming from:",
            DEFAULT_RESUME
        )

        checkpoint = torch.load(
            DEFAULT_RESUME,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        try:
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )
        except Exception:
            print(
                "Optimizer state not restored."
            )

        start_epoch = (
            int(
                checkpoint.get(
                    "epoch",
                    0,
                )
            )
            +
            1
        )

    best_score = -1.0

    for offset in range(
        epochs
    ):

        epoch = (
            start_epoch
            +
            offset
        )

        train_loss = train_one_epoch(
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
            f"Epoch {epoch}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
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
            ]
        )

        score = (
            validation[
                "pancreas_dice"
            ]
            +
            3.0
            *
            validation[
                "cancer_dice"
            ]
        )

        save_checkpoint(
            CHECKPOINT_DIR
            /
            "pancreas_tumor_fast_last.pt",
            model,
            optimizer,
            epoch,
            validation,
        )

        if score > best_score:

            best_score = score

            best_path = (
                CHECKPOINT_DIR
                /
                "pancreas_tumor_fast_best.pt"
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                validation,
            )

            print(
                "Saved best checkpoint:",
                best_path
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
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )