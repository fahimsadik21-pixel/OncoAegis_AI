"""
C-NMC 2019 ALL vs HEM classifier training.

Training:
    fold_0 + fold_1

Validation:
    fold_2

Class 0 = HEM
Class 1 = ALL

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
)

from app.data.hematology.cnmc2019_loader import (
    CNMC2019Dataset,
)

from app.imaging.hematology.cnmc_preprocessing import (
    preprocess_cnmc_image,
)

from app.models.hematology.cnmc_classifier import (
    CNMCClassifier,
)


SEED = 42

CHECKPOINT_DIR = Path(
    "checkpoints/cnmc2019"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    /
    "cnmc2019_classifier_best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    /
    "cnmc2019_classifier_last.pt"
)


class ProcessedCNMCDataset(Dataset):

    def __init__(
        self,
        folds,
        training: bool,
    ):

        self.base = CNMC2019Dataset(
            folds=folds
        )

        self.training = training

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

        image = sample[
            "image"
        ]

        if self.training:

            if random.random() < 0.5:
                image = torch.flip(
                    image,
                    dims=[2],
                )

            if random.random() < 0.5:
                image = torch.flip(
                    image,
                    dims=[1],
                )

            k = random.randint(
                0,
                3,
            )

            if k:
                image = torch.rot90(
                    image,
                    k=k,
                    dims=[
                        1,
                        2,
                    ],
                )

        image = preprocess_cnmc_image(
            image
        )

        return {
            "image":
                image,

            "label":
                torch.tensor(
                    sample[
                        "label"
                    ],
                    dtype=torch.long,
                ),

            "case":
                sample[
                    "case"
                ],
        }


def confusion_metrics(
    targets,
    predictions,
):

    targets = torch.cat(
        targets
    )

    predictions = torch.cat(
        predictions
    )

    tp = int(
        (
            (targets == 1)
            &
            (predictions == 1)
        ).sum()
    )

    tn = int(
        (
            (targets == 0)
            &
            (predictions == 0)
        ).sum()
    )

    fp = int(
        (
            (targets == 0)
            &
            (predictions == 1)
        ).sum()
    )

    fn = int(
        (
            (targets == 1)
            &
            (predictions == 0)
        ).sum()
    )

    accuracy = (
        (tp + tn)
        /
        max(
            tp + tn + fp + fn,
            1,
        )
    )

    sensitivity = (
        tp
        /
        max(
            tp + fn,
            1,
        )
    )

    specificity = (
        tn
        /
        max(
            tn + fp,
            1,
        )
    )

    balanced_accuracy = (
        sensitivity
        +
        specificity
    ) / 2.0

    precision = (
        tp
        /
        max(
            tp + fp,
            1,
        )
    )

    f1 = (
        2.0
        *
        precision
        *
        sensitivity
        /
        max(
            precision
            +
            sensitivity,
            1e-8,
        )
    )

    return {
        "accuracy":
            accuracy,

        "sensitivity_all":
            sensitivity,

        "specificity_hem":
            specificity,

        "balanced_accuracy":
            balanced_accuracy,

        "f1_all":
            f1,

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,
    }


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0

    targets = []
    predictions = []

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

                loss = criterion(
                    logits,
                    labels,
                )

            if training:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            prediction = torch.argmax(
                logits,
                dim=1,
            )

            targets.append(
                labels.detach().cpu()
            )

            predictions.append(
                prediction.detach().cpu()
            )

            total_loss += float(
                loss.item()
            )

            batches += 1

    metrics = confusion_metrics(
        targets,
        predictions,
    )

    metrics[
        "loss"
    ] = (
        total_loss
        /
        max(
            batches,
            1,
        )
    )

    return metrics


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
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
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

    train_dataset = ProcessedCNMCDataset(
        folds=(
            0,
            1,
        ),
        training=True,
    )

    validation_dataset = ProcessedCNMCDataset(
        folds=(
            2,
        ),
        training=False,
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

    train_labels = [
        int(
            sample[
                "label"
            ]
        )
        for sample
        in train_dataset.base.samples
    ]

    hem_count = train_labels.count(
        0
    )

    all_count = train_labels.count(
        1
    )

    total = (
        hem_count
        +
        all_count
    )

    class_weights = torch.tensor(
        [
            total
            /
            (
                2.0
                *
                hem_count
            ),

            total
            /
            (
                2.0
                *
                all_count
            ),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Training HEM:",
        hem_count
    )

    print(
        "Training ALL:",
        all_count
    )

    print(
        "Class weights:",
        class_weights.tolist()
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
        persistent_workers=True,
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
        persistent_workers=True,
    )

    model = CNMCClassifier().to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
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

    best_balanced_accuracy = -1.0
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

        best_balanced_accuracy = float(
            checkpoint.get(
                "best_balanced_accuracy",
                -1.0,
            )
        )

    for epoch in range(
        start_epoch,
        start_epoch
        +
        args.epochs,
    ):

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            training=True,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            training=False,
        )

        print(
            f"Epoch {epoch} | "
            f"TrainLoss {train_metrics['loss']:.4f} | "
            f"TrainAcc {train_metrics['accuracy']:.4f} | "
            f"ValLoss {validation_metrics['loss']:.4f} | "
            f"ValAcc {validation_metrics['accuracy']:.4f} | "
            f"BalAcc {validation_metrics['balanced_accuracy']:.4f} | "
            f"SensALL {validation_metrics['sensitivity_all']:.4f} | "
            f"SpecHEM {validation_metrics['specificity_hem']:.4f} | "
            f"F1ALL {validation_metrics['f1_all']:.4f}"
        )

        current_balanced_accuracy = (
            validation_metrics[
                "balanced_accuracy"
            ]
        )

        checkpoint = {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_metrics":
                validation_metrics,

            "best_balanced_accuracy":
                max(
                    best_balanced_accuracy,
                    current_balanced_accuracy,
                ),

            "model_name":
                "C-NMC 2019 ALL Cell Classifier",

            "model_version":
                f"research-baseline-epoch-{epoch}",

            "classes": {
                0: "HEM",
                1: "ALL",
            },

            "input_size": (
                224,
                224,
            ),

            "training_folds": [
                0,
                1,
            ],

            "validation_fold":
                2,
        }

        torch.save(
            checkpoint,
            LAST_CHECKPOINT,
        )

        if (
            current_balanced_accuracy
            >
            best_balanced_accuracy
        ):

            best_balanced_accuracy = (
                current_balanced_accuracy
            )

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
        "Best balanced accuracy:",
        best_balanced_accuracy
    )


if __name__ == "__main__":
    main()