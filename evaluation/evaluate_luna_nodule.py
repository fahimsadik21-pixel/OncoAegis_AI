"""Evaluate the LUNA16 annotated-nodule slice classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.metrics import classification_report
from src.data.loaders import create_luna_nodule_loaders
from src.models.luna_models import LUNANoduleClassifier
from training.segmentation import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed/luna16"))
    parser.add_argument("--annotations", type=Path, default=Path("datasets/universal_ct/luna16/annotations.csv"))
    parser.add_argument("--raw-ct-dir", type=Path, default=Path("datasets/universal_ct/luna16/subset1"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-slices", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = LUNANoduleClassifier(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loaders = create_luna_nodule_loaders(
        args.data_dir,
        args.annotations,
        args.raw_ct_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        max_validation_slices=args.max_validation_slices,
    )
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loaders.validation:
            logits.append(model(images.to(device)).cpu())
            labels.append(targets)
    report = classification_report(
        torch.cat(logits),
        torch.cat(labels),
        class_names=("no_annotated_nodule", "annotated_nodule"),
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
