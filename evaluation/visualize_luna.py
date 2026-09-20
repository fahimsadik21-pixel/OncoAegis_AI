"""Save LUNA16 lung-segmentation predictions for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.loaders import create_luna16_loaders
from src.models.unet import UNet
from training.segmentation import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed/luna16"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/luna16_predictions"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loaders = create_luna16_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.inference_mode():
        for images, masks in loaders.validation:
            predictions = (torch.sigmoid(model(images.to(device))) >= 0.5).cpu()
            for index in range(images.shape[0]):
                if saved >= args.count:
                    break
                figure, axes = plt.subplots(1, 4, figsize=(14, 4))
                axes[0].imshow(images[index, 0], cmap="gray")
                axes[0].set_title("CT slice")
                axes[1].imshow(masks[index, 0], cmap="gray")
                axes[1].set_title("Lung ground truth")
                axes[2].imshow(predictions[index, 0], cmap="gray")
                axes[2].set_title("Lung prediction")
                axes[3].imshow(images[index, 0], cmap="gray")
                axes[3].imshow(predictions[index, 0], cmap="Reds", alpha=0.35)
                axes[3].set_title("Overlay")
                for axis in axes:
                    axis.axis("off")
                figure.tight_layout()
                figure.savefig(args.output_dir / f"{saved:03d}.png", dpi=140, bbox_inches="tight")
                plt.close(figure)
                saved += 1
            if saved >= args.count:
                break
    print(f"Saved {saved} LUNA16 prediction figures to {args.output_dir}")


if __name__ == "__main__":
    main()
