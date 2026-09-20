"""Train LUNA16 nodule-vs-no-nodule slice classification."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import create_luna_nodule_loaders
from src.models.luna_models import LUNANoduleClassifier
from training.classification import fit_classification
from training.segmentation import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed/luna16"))
    parser.add_argument("--annotations", type=Path, default=Path("datasets/universal_ct/luna16/annotations.csv"))
    parser.add_argument("--raw-ct-dir", type=Path, default=Path("datasets/universal_ct/luna16/subset1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/luna16_nodule"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-slices", type=int, default=None)
    parser.add_argument("--max-validation-slices", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    loaders = create_luna_nodule_loaders(
        args.data_dir,
        args.annotations,
        args.raw_ct_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        negative_ratio=args.negative_ratio,
        max_train_slices=args.max_train_slices,
        max_validation_slices=args.max_validation_slices,
    )
    print(f"Device: {device}")
    print(f"Train cases: {len(loaders.train_case_ids)} | validation cases: {len(loaders.validation_case_ids)}")
    print(f"Train slices used: {len(loaders.train.dataset)} | validation slices used: {len(loaders.validation.dataset)}")
    model = LUNANoduleClassifier(num_classes=2, base_channels=args.base_channels)
    fit_classification(
        model,
        loaders.train,
        loaders.validation,
        device=device,
        num_classes=2,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="luna16_nodule_classifier",
        metadata={
            "dataset": "LUNA16",
            "task": "annotated_nodule_vs_no_annotated_nodule",
            "train_case_ids": list(loaders.train_case_ids),
            "validation_case_ids": list(loaders.validation_case_ids),
            "seed": args.seed,
        },
        use_amp=not args.no_amp,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
