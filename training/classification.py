"""Reusable PyTorch loop for binary or multi-class slice classification."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from evaluation.metrics import classification_metrics
from training.segmentation import _amp_context, _make_scaler


def run_classification_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "samples": 0.0}
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, use_amp):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        metrics = classification_metrics(logits.detach(), labels, num_classes=num_classes)
        count = metrics["samples"]
        totals["loss"] += float(loss.detach().cpu()) * count
        totals["accuracy"] += metrics["accuracy"] * count
        totals["macro_f1"] += metrics["macro_f1"] * count
        totals["samples"] += count
    if totals["samples"] == 0:
        raise ValueError("The data loader yielded no batches")
    for key in ("loss", "accuracy", "macro_f1"):
        totals[key] /= totals["samples"]
    return totals


def fit_classification(
    model: torch.nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    *,
    device: torch.device,
    num_classes: int,
    epochs: int,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_name: str = "classification",
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
    best_f1 = float("-inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_classification_epoch(model, train_loader, device=device, num_classes=num_classes, optimizer=optimizer, scaler=scaler, use_amp=use_amp, max_batches=max_batches)
        with torch.no_grad():
            validation_metrics = run_classification_epoch(model, validation_loader, device=device, num_classes=num_classes, use_amp=use_amp, max_batches=max_batches)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
            f"val accuracy {validation_metrics['accuracy']:.4f} | val macro-F1 {validation_metrics['macro_f1']:.4f}"
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": {
                "num_classes": getattr(model, "num_classes", num_classes),
                "base_channels": getattr(model, "base_channels", 16),
            },
            "epoch": epoch,
            "metrics": record,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_last.pt")
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            torch.save(checkpoint, checkpoint_root / f"{checkpoint_name}_best.pt")
    return history
