from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.data.ultrasound.tn3k_dataset import (
    build_tn3k_datasets,
)

from app.models.ultrasound.tn3k_unet import (
    TN3KUNet,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "tn3k_segmentation"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "tn3k_unet_best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "tn3k_unet_last.pt"
)


def seed_everything(
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        predictions * targets
    ).sum(dim=1)

    denominator = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return dice.mean()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

    def forward(
        self,
        logits,
        targets,
    ):

        bce = self.bce(
            logits,
            targets,
        )

        dice = soft_dice_loss(
            logits,
            targets,
        )

        return (
            0.5 * bce
            + 0.5 * dice
        )


def run_epoch(
    *,
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    training: bool,
    scaler,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        batch_size = images.size(0)

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    device.type == "cuda"
                ),
            ):

                logits = model(images)

                loss = loss_fn(
                    logits,
                    masks,
                )

            if training:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

        with torch.no_grad():

            dice = dice_score_from_logits(
                logits,
                masks,
            )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_dice += (
            dice.item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "loss": (
            total_loss
            / max(total_samples, 1)
        ),
        "dice": (
            total_dice
            / max(total_samples, 1)
        ),
    }


def save_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    fold: int,
    image_size: int,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_dice":
                best_dice,

            "fold":
                fold,

            "image_size":
                image_size,

            "model_name":
                "TN3KUNet",

            "dataset":
                "TN3K",
        },
        path,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "DEVICE:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    train_ds, val_ds, _ = (
        build_tn3k_datasets(
            fold=args.fold,
            image_size=args.image_size,
        )
    )

    print(
        "TRAIN:",
        len(train_ds),
    )

    print(
        "VAL:",
        len(val_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = TN3KUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    loss_fn = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    start_epoch = 1
    best_val_dice = -1.0

    if (
        args.resume
        and LAST_CHECKPOINT.exists()
    ):

        print(
            "RESUMING:",
            LAST_CHECKPOINT,
        )

        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location=device,
            weights_only=False,
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

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_dice = (
            checkpoint[
                "best_val_dice"
            ]
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            training=True,
            scaler=scaler,
        )

        with torch.no_grad():

            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                training=False,
                scaler=scaler,
            )

        scheduler.step(
            val_metrics["dice"]
        )

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"EPOCH {epoch:02d} | "
            f"LR {lr:.6f} | "
            f"TRAIN LOSS "
            f"{train_metrics['loss']:.6f} | "
            f"TRAIN DICE "
            f"{train_metrics['dice']:.6f} | "
            f"VAL LOSS "
            f"{val_metrics['loss']:.6f} | "
            f"VAL DICE "
            f"{val_metrics['dice']:.6f}"
        )

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_dice=best_val_dice,
            fold=args.fold,
            image_size=args.image_size,
        )

        if (
            val_metrics["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_metrics["dice"]
            )

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_dice=best_val_dice,
                fold=args.fold,
                image_size=args.image_size,
            )

            print(
                "BEST CHECKPOINT SAVED:",
                BEST_CHECKPOINT,
                "DICE:",
                f"{best_val_dice:.6f}",
            )

    print()
    print(
        "TRAINING COMPLETE"
    )

    print(
        "BEST VAL DICE:",
        f"{best_val_dice:.6f}",
    )

    print(
        "BEST CHECKPOINT:",
        BEST_CHECKPOINT,
    )


if __name__ == "__main__":
    main()