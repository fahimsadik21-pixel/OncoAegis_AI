"""Save qualitative BUSI validation predictions as PNG figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.busi_dataset import BUSI_CLASSES
from src.data.loaders import create_busi_loaders
from src.models.busi_models import BUSIMultiTaskUNet
from training.segmentation import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/busi_predictions"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = BUSIMultiTaskUNet(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loaders = create_busi_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for batch in loaders.validation:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            predicted_masks = (torch.sigmoid(outputs["segmentation_logits"]) >= 0.5).cpu()
            predicted_labels = outputs["classification_logits"].argmax(dim=1).cpu()
            for index in range(images.shape[0]):
                if saved >= args.count:
                    break
                image = batch["image"][index, 0].numpy()
                ground_truth = batch["mask"][index, 0].numpy()
                prediction = predicted_masks[index, 0].numpy()
                sample_id = str(batch["sample_id"][index]).replace("/", "_").replace("\\", "_")
                true_name = str(batch["class_name"][index])
                predicted_name = BUSI_CLASSES[int(predicted_labels[index])]
                figure, axes = plt.subplots(1, 4, figsize=(14, 4))
                axes[0].imshow(image, cmap="gray")
                axes[0].set_title("Ultrasound")
                axes[1].imshow(ground_truth, cmap="gray")
                axes[1].set_title("Ground truth")
                axes[2].imshow(prediction, cmap="gray")
                axes[2].set_title("Predicted mask")
                axes[3].imshow(image, cmap="gray")
                axes[3].imshow(prediction, cmap="Reds", alpha=0.35)
                axes[3].set_title("Overlay")
                for axis in axes:
                    axis.axis("off")
                figure.suptitle(f"GT: {true_name} | Pred: {predicted_name}")
                figure.tight_layout()
                figure.savefig(args.output_dir / f"{saved:03d}_{sample_id}.png", dpi=140, bbox_inches="tight")
                plt.close(figure)
                saved += 1
            if saved >= args.count:
                break
    print(f"Saved {saved} prediction figures to {args.output_dir}")


if __name__ == "__main__":
    main()
