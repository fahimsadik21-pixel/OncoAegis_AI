"""Evaluate a saved LUNA16 U-Net checkpoint on its deterministic validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.loaders import create_luna16_loaders
from src.models.unet import UNet
from training.segmentation import run_segmentation_epoch, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed/luna16"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-slices", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("model_config", {})
    model = UNet(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loaders = create_luna16_loaders(args.data_dir, batch_size=args.batch_size, image_size=(args.image_size, args.image_size), seed=args.seed, pin_memory=device.type == "cuda", max_validation_slices=args.max_validation_slices)
    with torch.no_grad():
        metrics = run_segmentation_epoch(model, loaders.validation, device=device)
    print(metrics)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
