r"""Train the LUNA16 2D lung-segmentation baseline.

Example smoke run:
    .venv\Scripts\python.exe -m training.train_luna --epochs 1 --max-train-slices 32 --max-validation-slices 16 --max-batches 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import create_luna16_loaders
from src.models.unet import UNet
from training.segmentation import fit_segmentation, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed/luna16"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/luna16"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-slices", type=int, default=None)
    parser.add_argument("--max-validation-slices", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for a quick smoke run")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--positive-only", action="store_true", help="Exclude slices with an empty lung mask")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    loaders = create_luna16_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size) if args.image_size > 0 else None,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        include_empty=not args.positive_only,
        max_train_slices=args.max_train_slices,
        max_validation_slices=args.max_validation_slices,
    )
    print(f"Device: {device}")
    print(f"Train cases: {len(loaders.train_case_ids)} | validation cases: {len(loaders.validation_case_ids)}")
    print(f"Train slices used: {len(loaders.train.dataset)} | validation slices used: {len(loaders.validation.dataset)}")

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels)
    fit_segmentation(
        model,
        loaders.train,
        loaders.validation,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="luna16_unet",
        metadata={
            "dataset": "LUNA16",
            "train_case_ids": list(loaders.train_case_ids),
            "validation_case_ids": list(loaders.validation_case_ids),
            "image_size": args.image_size,
            "seed": args.seed,
        },
        use_amp=not args.no_amp,
        max_batches=args.max_batches,
    )
    print(f"Checkpoints saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
