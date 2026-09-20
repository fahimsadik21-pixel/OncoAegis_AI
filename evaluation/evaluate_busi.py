"""Evaluate a saved BUSI multi-task checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.loaders import create_busi_loaders
from src.models.busi_models import BUSIMultiTaskUNet
from evaluation.busi_report import evaluate_busi_model
from training.segmentation import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BUSIMultiTaskUNet(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loaders = create_busi_loaders(args.data_dir, batch_size=args.batch_size, image_size=(args.image_size, args.image_size), seed=args.seed, pin_memory=device.type == "cuda", max_validation_samples=args.max_validation_samples)
    metrics = evaluate_busi_model(model, loaders.validation, device=device)
    import json

    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
