r"""Train the BUSI multi-task segmentation/classification baseline.

Example smoke run:
    .venv\Scripts\python.exe -m training.train_busi --epochs 1 --max-train-samples 12 --max-validation-samples 6 --max-batches 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.data.busi_dataset import BUSI_CLASSES
from src.data.loaders import create_busi_loaders
from src.models.busi_models import BUSIMultiTaskUNet
from training.multitask import fit_multitask
from training.segmentation import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/busi"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for a quick smoke run")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    loaders = create_busi_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size) if args.image_size > 0 else None,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )
    print(f"Device: {device}")
    print(f"Train samples available: {len(loaders.train_samples)} | validation samples available: {len(loaders.validation_samples)}")
    print(f"Train samples used: {len(loaders.train.dataset)} | validation samples used: {len(loaders.validation.dataset)}")
    print(f"Classes: {', '.join(BUSI_CLASSES)}")

    model = BUSIMultiTaskUNet(num_classes=len(BUSI_CLASSES), base_channels=args.base_channels)
    fit_multitask(
        model,
        loaders.train,
        loaders.validation,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="busi_multitask_unet",
        metadata={
            "dataset": "BUSI",
            "classes": list(BUSI_CLASSES),
            "train_sample_ids": [sample.sample_id for sample in loaders.train_samples],
            "validation_sample_ids": [sample.sample_id for sample in loaders.validation_samples],
            "image_size": args.image_size,
            "seed": args.seed,
        },
        use_amp=not args.no_amp,
        max_batches=args.max_batches,
    )
    print(f"Checkpoints saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
