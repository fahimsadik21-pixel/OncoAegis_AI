"""Reusable BUSI segmentation + classification training loop."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from evaluation.metrics import classification_metrics, segmentation_metrics
from src.models.losses import combined_busi_loss
from training.segmentation import _amp_context, _make_scaler


def run_multitask_epoch(
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
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "positive_dice": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "samples": 0.0, "positive_samples": 0.0}

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, use_amp):
            outputs = model(images)
            loss = combined_busi_loss(outputs["segmentation_logits"], masks, outputs["classification_logits"], labels)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        seg = segmentation_metrics(outputs["segmentation_logits"].detach(), masks)
        cls = classification_metrics(outputs["classification_logits"].detach(), labels)
        count = seg["samples"]
        totals["loss"] += float(loss.detach().cpu()) * count
        for key in ("dice", "iou", "positive_dice", "accuracy", "macro_f1"):
            totals[key] += (seg if key in seg else cls)[key] * count
        totals["samples"] += count
        totals["positive_samples"] += seg["positive_samples"]

    if totals["samples"] == 0:
        raise ValueError("The data loader yielded no batches")
    for key in ("loss", "dice", "iou", "positive_dice", "accuracy", "macro_f1"):
        totals[key] /= totals["samples"]
    return totals


def fit_multitask(
    model: torch.nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_name: str = "busi_multitask",
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
    best_score = float("-inf")
    best_dice = float("-inf")
    best_accuracy = float("-inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_multitask_epoch(model, train_loader, device=device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, max_batches=max_batches)
        with torch.no_grad():
            validation_metrics = run_multitask_epoch(model, validation_loader, device=device, use_amp=use_amp, max_batches=max_batches)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val dice {validation_metrics['dice']:.4f} | val accuracy {validation_metrics['accuracy']:.4f}"
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": {"num_classes": getattr(model, "num_classes", 3), "base_channels": getattr(model, "base_channels", 16)},
            "epoch": epoch,
            "metrics": record,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_last.pt")
        score = validation_metrics["dice"] + validation_metrics["accuracy"]
        if score > best_score:
            best_score = score
            torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_best.pt")
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_best_segmentation.pt")
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_best_classification.pt")
    return history
