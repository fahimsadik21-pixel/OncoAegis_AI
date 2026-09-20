"""Reusable PyTorch loop for binary image segmentation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from evaluation.metrics import segmentation_metrics
from src.models.losses import combined_segmentation_loss


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def _make_scaler(device: torch.device, enabled: bool):
    amp_enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def run_segmentation_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "positive_dice": 0.0, "samples": 0.0, "positive_samples": 0.0}

    for batch_index, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, use_amp):
            logits = model(images)
            loss = combined_segmentation_loss(logits, masks)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        metrics = segmentation_metrics(logits.detach(), masks)
        count = metrics["samples"]
        totals["loss"] += float(loss.detach().cpu()) * count
        for key in ("dice", "iou", "positive_dice"):
            totals[key] += metrics[key] * count
        totals["samples"] += count
        totals["positive_samples"] += metrics["positive_samples"]

    if totals["samples"] == 0:
        raise ValueError("The data loader yielded no batches")
    for key in ("loss", "dice", "iou", "positive_dice"):
        totals[key] /= totals["samples"]
    return totals


def fit_segmentation(
    model: torch.nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_name: str = "segmentation",
    metadata: dict[str, Any] | None = None,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = _make_scaler(device, use_amp)
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_dice = float("-inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_segmentation_epoch(model, train_loader, device=device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, max_batches=max_batches)
        with torch.no_grad():
            validation_metrics = run_segmentation_epoch(model, validation_loader, device=device, use_amp=use_amp, max_batches=max_batches)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val dice {validation_metrics['dice']:.4f} | val IoU {validation_metrics['iou']:.4f}"
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": {
                "in_channels": getattr(model, "in_channels", 1),
                "out_channels": getattr(model, "out_channels", 1),
                "base_channels": getattr(model, "base_channels", 16),
            },
            "epoch": epoch,
            "metrics": record,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_last.pt")
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_best.pt")
    return history
