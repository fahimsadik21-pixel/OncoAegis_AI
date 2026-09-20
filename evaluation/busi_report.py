"""Full BUSI validation report, including class-wise classification metrics."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from evaluation.metrics import classification_report, segmentation_metrics


def evaluate_busi_model(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    segmentation_totals = {"dice": 0.0, "iou": 0.0, "positive_dice": 0.0, "samples": 0.0, "positive_samples": 0.0}
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    from src.models.losses import combined_busi_loss

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            loss = combined_busi_loss(outputs["segmentation_logits"], masks, outputs["classification_logits"], labels)
            losses.append(float(loss.detach().cpu()))
            segmentation = segmentation_metrics(outputs["segmentation_logits"], masks)
            count = segmentation["samples"]
            for key in ("dice", "iou", "positive_dice"):
                segmentation_totals[key] += segmentation[key] * count
            segmentation_totals["samples"] += count
            segmentation_totals["positive_samples"] += segmentation["positive_samples"]
            all_logits.append(outputs["classification_logits"].detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_labels or segmentation_totals["samples"] == 0:
        raise ValueError("The BUSI validation loader yielded no samples")
    for key in ("dice", "iou", "positive_dice"):
        segmentation_totals[key] /= segmentation_totals["samples"]
    report = classification_report(torch.cat(all_logits), torch.cat(all_labels))
    return {
        "loss": sum(losses) / len(losses),
        "segmentation": segmentation_totals,
        "classification": report,
    }
