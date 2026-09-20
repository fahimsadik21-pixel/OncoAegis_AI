"""
IRCADb01 Stage-2 Liver Tumor Training

Deep Learning:
    3D U-Net

Classes:
    0 = background
    1 = liver
    2 = tumor

Uses liver-centered ROI preprocessing.

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from app.data.ct.ircadb01_liver_loader import (
    IRCADLiverDataset,
)

from app.imaging.ct.liver_preprocessing import (
    preprocess_liver_volume,
)

from app.models.ct.liver_tumor_unet import (
    LiverTumor3DUNet,
)


_PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


CHECKPOINT_DIR = (
    _PROJECT_ROOT
    / "checkpoints"
    / "liver_tumor_stage2"
)


BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "liver_tumor_stage2_best.pt"
)


LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "liver_tumor_stage2_last.pt"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class LiverTumorROIDataset(
    Dataset
):

    def __init__(
        self,
    ):

        self.base_dataset = (
            IRCADLiverDataset()
        )

    def __len__(
        self,
    ):

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.base_dataset[
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
            preprocess_liver_volume(
                image=image,
                label=label,
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

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

        target_one_hot = (
            F.one_hot(
                target.long(),
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

        # Ignore background in Dice.
        probabilities = (
            probabilities[:, 1:]
        )

        target_one_hot = (
            target_one_hot[:, 1:]
        )

        intersection = (
            probabilities
            *
            target_one_hot
        ).sum(
            dim=(
                2,
                3,
                4,
            )
        )

        denominator = (
            probabilities.sum(
                dim=(
                    2,
                    3,
                    4,
                )
            )
            +
            target_one_hot.sum(
                dim=(
                    2,
                    3,
                    4,
                )
            )
        )

        dice = (
            2.0 * intersection
            +
            self.smooth
        ) / (
            denominator
            +
            self.smooth
        )

        return (
            1.0
            -
            dice.mean()
        )


class TumorWeightedLoss(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.dice = (
            ForegroundDiceLoss()
        )

        # Tumor class gets highest weight.
        self.register_buffer(
            "class_weights",
            torch.tensor(
                [
                    0.5,
                    1.5,
                    8.0,
                ],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits,
        target,
    ):

        ce_loss = (
            F.cross_entropy(
                logits,
                target,
                weight=self.class_weights,
            )
        )

        dice_loss = (
            self.dice(
                logits,
                target,
            )
        )

        return (
            ce_loss
            +
            dice_loss
        )


def tumor_dice_score(
    prediction,
    target,
):

    pred = (
        prediction == 2
    )

    truth = (
        target == 2
    )

    intersection = (
        pred
        &
        truth
    ).sum().float()

    denominator = (
        pred.sum().float()
        +
        truth.sum().float()
    )

    if denominator.item() == 0:

        return 1.0

    return float(
        (
            2.0
            *
            intersection
        )
        /
        denominator
    )


def liver_dice_score(
    prediction,
    target,
):

    pred = (
        prediction == 1
    )

    truth = (
        target == 1
    )

    intersection = (
        pred
        &
        truth
    ).sum().float()

    denominator = (
        pred.sum().float()
        +
        truth.sum().float()
    )

    if denominator.item() == 0:

        return 1.0

    return float(
        (
            2.0
            *
            intersection
        )
        /
        denominator
    )


def train_liver_tumor(
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
):

    print(
        f"Using device: {DEVICE}"
    )

    dataset = (
        LiverTumorROIDataset()
    )

    train_size = max(
        1,
        int(
            len(dataset)
            *
            0.8
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

    (
        train_dataset,
        val_dataset,
    ) = random_split(

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

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,
    )

    model = (
        LiverTumor3DUNet(
            in_channels=1,
            out_channels=3,
        )
        .to(
            DEVICE
        )
    )

    criterion = (
        TumorWeightedLoss()
        .to(
            DEVICE
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
    )

    use_amp = (
        DEVICE.type
        ==
        "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        epochs
    ):

        model.train()

        train_loss_total = 0.0

        for (
            batch_index,
            batch,
        ) in enumerate(
            train_loader,
            start=1,
        ):

            images = (
                batch["image"]
                .to(
                    DEVICE
                )
            )

            labels = (
                batch["label"]
                .to(
                    DEVICE
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            train_loss_total += (
                float(
                    loss.item()
                )
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train batch "
                f"{batch_index}/"
                f"{len(train_loader)} "
                f"Loss: "
                f"{loss.item():.4f}"
            )

            del images
            del labels
            del outputs
            del loss

            if DEVICE.type == "cuda":

                torch.cuda.empty_cache()

        train_loss = (
            train_loss_total
            /
            max(
                1,
                len(
                    train_loader
                ),
            )
        )

        model.eval()

        val_loss_total = 0.0

        liver_dice_total = 0.0
        tumor_dice_total = 0.0

        with torch.no_grad():

            for batch in val_loader:

                images = (
                    batch["image"]
                    .to(
                        DEVICE
                    )
                )

                labels = (
                    batch["label"]
                    .to(
                        DEVICE
                    )
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

                prediction = (
                    torch.argmax(
                        outputs,
                        dim=1,
                    )
                )

                liver_dice_total += (
                    liver_dice_score(
                        prediction,
                        labels,
                    )
                )

                tumor_dice_total += (
                    tumor_dice_score(
                        prediction,
                        labels,
                    )
                )

                val_loss_total += (
                    float(
                        loss.item()
                    )
                )

                del images
                del labels
                del outputs
                del prediction
                del loss

                if DEVICE.type == "cuda":

                    torch.cuda.empty_cache()

        val_loss = (
            val_loss_total
            /
            max(
                1,
                len(
                    val_loader
                ),
            )
        )

        liver_dice = (
            liver_dice_total
            /
            max(
                1,
                len(
                    val_loader
                ),
            )
        )

        tumor_dice = (
            tumor_dice_total
            /
            max(
                1,
                len(
                    val_loader
                ),
            )
        )

        print()

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Liver Dice: "
            f"{liver_dice:.4f}"
        )

        print(
            f"Validation Tumor Dice: "
            f"{tumor_dice:.4f}"
        )

        torch.save(
            model.state_dict(),
            LAST_MODEL_PATH,
        )

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH,
            )

            print(
                "Saved best "
                "Stage-2 checkpoint"
            )

        print()

    print(
        "Stage-2 liver tumor "
        "training completed"
    )

    print(
        f"Best checkpoint: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Last checkpoint: "
        f"{LAST_MODEL_PATH}"
    )


if __name__ == "__main__":

    train_liver_tumor()