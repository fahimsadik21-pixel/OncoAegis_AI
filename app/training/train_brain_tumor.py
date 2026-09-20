"""
MSD Brain Tumor 3D U-Net Training Pipeline

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split,
)

from torch.cuda.amp import (
    autocast,
    GradScaler,
)


from app.data.mri.msd_brain_loader import (
    MSDBrainDataset
)

from app.models.mri.brain_tumor_unet import (
    BrainTumor3DUNet
)



_PROJECT_ROOT = Path(__file__).resolve().parents[2]


CHECKPOINT_DIR = (
    _PROJECT_ROOT
    /
    "checkpoints"
    /
    "msd_brain_tumor"
)


BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    /
    "msd_brain_tumor_best.pt"
)



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ============================
# Dice Loss
# ============================


class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(
        self,
        prediction,
        target
    ):

        smooth = 1e-5


        prediction = torch.softmax(
            prediction,
            dim=1
        )


        target_one_hot = torch.nn.functional.one_hot(
            target.long(),
            num_classes=prediction.shape[1]
        )


        target_one_hot = target_one_hot.permute(
            0,
            4,
            1,
            2,
            3
        ).float()



        intersection = (
            prediction *
            target_one_hot
        ).sum(
            dim=(2,3,4)
        )


        denominator = (
            prediction +
            target_one_hot
        ).sum(
            dim=(2,3,4)
        )


        dice = (
            (2 * intersection + smooth)
            /
            (denominator + smooth)
        )


        return 1 - dice.mean()



# ============================
# Combined Loss
# ============================


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice = DiceLoss()

        self.ce = nn.CrossEntropyLoss()



    def forward(
        self,
        output,
        target
    ):

        return (
            self.dice(
                output,
                target
            )
            +
            self.ce(
                output,
                target
            )
        )



# ============================
# Training
# ============================


def train_brain_tumor(
    epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
):


    print(
        f"Using device: {DEVICE}"
    )


    dataset = MSDBrainDataset(
        target_size=128
    )


    train_size = int(
        len(dataset) * 0.8
    )


    val_size = (
        len(dataset)
        -
        train_size
    )


    train_dataset, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=torch.Generator().manual_seed(42)
    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )



    model = BrainTumor3DUNet(
        in_channels=4,
        out_channels=4
    )


    model.to(
        DEVICE
    )



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )



    criterion = CombinedLoss()



    scaler = GradScaler(
        enabled=(
            DEVICE == "cuda"
        )
    )



    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



    best_loss = float(
        "inf"
    )



    for epoch in range(
        epochs
    ):


        model.train()


        train_loss = 0



        for batch in train_loader:


            images = batch["image"].to(
                DEVICE,
                non_blocking=True
            )


            labels = batch["label"].to(
                DEVICE,
                non_blocking=True
            )



            optimizer.zero_grad()



            with autocast(
                enabled=(
                    DEVICE == "cuda"
                )
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



            train_loss += loss.item()



            if DEVICE == "cuda":

                torch.cuda.empty_cache()



        train_loss /= len(
            train_loader
        )



        # ====================
        # Validation
        # ====================


        model.eval()


        val_loss = 0



        with torch.no_grad():


            for batch in val_loader:


                images = batch["image"].to(
                    DEVICE
                )


                labels = batch["label"].to(
                    DEVICE
                )



                with autocast(
                    enabled=(
                        DEVICE == "cuda"
                    )
                ):


                    outputs = model(
                        images
                    )


                    loss = criterion(
                        outputs,
                        labels
                    )



                val_loss += loss.item()



        val_loss /= len(
            val_loader
        )



        print(
            f"""
Epoch {epoch+1}/{epochs}

Train Loss:
{train_loss:.4f}

Validation Loss:
{val_loss:.4f}
"""
        )



        if val_loss < best_loss:


            best_loss = val_loss



            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )


            print(
                "Saved best checkpoint"
            )



        if DEVICE == "cuda":

            torch.cuda.empty_cache()



    print(
        "Training completed"
    )


    print(
        f"Best model saved: {BEST_MODEL_PATH}"
    )



if __name__ == "__main__":

    train_brain_tumor()