"""
IRCADb01 Liver Locator Training

Stage 1:
    Full abdominal CT
        -> coarse liver-region segmentation

Classes:
    0 = background
    1 = liver region

Tumor voxels are included in the liver region.

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from app.data.ct.ircadb01_liver_loader import IRCADLiverDataset
from app.models.ct.liver_locator_unet import LiverLocator3DUNet


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = (
    _PROJECT_ROOT
    / "checkpoints"
    / "liver_locator"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "liver_locator_best.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "liver_locator_last.pt"
)

TARGET_SIZE = (
    128,
    128,
    128,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class LiverLocatorDataset(Dataset):

    def __init__(self):

        self.base_dataset = IRCADLiverDataset()

    def __len__(self):

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

        image = sample[
            "image"
        ].float()

        label = sample[
            "label"
        ]

        # Liver locator target:
        # both normal liver and tumor are treated
        # as part of the liver region.
        liver_target = (
            label > 0
        ).long()

        image = image.unsqueeze(
            0
        )

        image = F.interpolate(
            image,
            size=TARGET_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        image = image.squeeze(
            0
        )

        liver_target = liver_target.float()

        liver_target = liver_target.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        liver_target = F.interpolate(
            liver_target,
            size=TARGET_SIZE,
            mode="nearest",
        )

        liver_target = liver_target.squeeze(
            0
        ).squeeze(
            0
        ).long()

        return {
            "image":
                image,

            "label":
                liver_target,

            "case":
                sample["case"],
        }


class BinaryDiceLoss(nn.Module):

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
        )[:, 1]

        target = (
            target == 1
        ).float()

        intersection = (
            probabilities
            *
            target
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            target.sum(
                dim=(1, 2, 3)
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


class CombinedLocatorLoss(nn.Module):

    def __init__(self):

        super().__init__()

        # Background is far more common than liver,
        # so liver gets higher weight.
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [
                    1.0,
                    4.0,
                ],
                dtype=torch.float32,
            )
        )

        self.dice = BinaryDiceLoss()

    def forward(
        self,
        logits,
        target,
    ):

        weight = self.ce.weight.to(
            logits.device
        )

        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=weight,
        )

        dice_loss = self.dice(
            logits,
            target,
        )

        return (
            ce_loss
            +
            dice_loss
        )


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
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
            2.0 * intersection
        )
        /
        denominator
    )


def train_liver_locator(
    epochs: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
):

    print(
        f"Using device: {DEVICE}"
    )

    dataset = LiverLocatorDataset()

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

    train_dataset, val_dataset = random_split(
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

    model = LiverLocator3DUNet(
        in_channels=1,
        out_channels=2,
    ).to(
        DEVICE
    )

    criterion = CombinedLocatorLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    use_amp = (
        DEVICE.type
        ==
        "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
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

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            images = batch[
                "image"
            ].to(
                DEVICE
            )

            labels = batch[
                "label"
            ].to(
                DEVICE
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

            train_loss_total += float(
                loss.item()
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train batch {batch_index}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
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
                len(train_loader),
            )
        )

        model.eval()

        val_loss_total = 0.0
        val_dice_total = 0.0

        with torch.no_grad():

            for batch in val_loader:

                images = batch[
                    "image"
                ].to(
                    DEVICE
                )

                labels = batch[
                    "label"
                ].to(
                    DEVICE
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

                prediction = torch.argmax(
                    outputs,
                    dim=1,
                )

                val_dice_total += dice_score(
                    prediction,
                    labels,
                )

                val_loss_total += float(
                    loss.item()
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
                len(val_loader),
            )
        )

        val_dice = (
            val_dice_total
            /
            max(
                1,
                len(val_loader),
            )
        )

        print()
        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Liver Dice: {val_dice:.4f}"
        )

        torch.save(
            model.state_dict(),
            LAST_MODEL_PATH,
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH,
            )

            print(
                "Saved best liver locator checkpoint"
            )

        print()

    print(
        "Liver locator training completed"
    )

    print(
        f"Best checkpoint: {BEST_MODEL_PATH}"
    )

    print(
        f"Last checkpoint: {LAST_MODEL_PATH}"
    )


if __name__ == "__main__":

    train_liver_locator()