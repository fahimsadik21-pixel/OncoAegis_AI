"""Simple dependency-free metrics for medical segmentation/classification."""

from __future__ import annotations

import torch


def _per_sample_dice(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    return (2.0 * intersection + smooth) / (prediction.sum(dim=1) + target.sum(dim=1) + smooth)


def _per_sample_iou(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    union = prediction.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= threshold).float()
    dice = _per_sample_dice(prediction, target).mean()
    iou = _per_sample_iou(prediction, target).mean()
    positive = target.flatten(1).sum(dim=1) > 0
    positive_dice = _per_sample_dice(prediction[positive], target[positive]).mean() if positive.any() else torch.tensor(0.0, device=target.device)
    return {
        "dice": float(dice.detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "positive_dice": float(positive_dice.detach().cpu()),
        "positive_samples": float(positive.sum().detach().cpu()),
        "samples": float(target.shape[0]),
    }


def classification_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> dict[str, float]:
    prediction = logits.argmax(dim=1)
    target = target.long()
    accuracy = (prediction == target).float().mean()
    f1_values: list[torch.Tensor] = []
    for class_index in range(num_classes):
        true_positive = ((prediction == class_index) & (target == class_index)).sum().float()
        false_positive = ((prediction == class_index) & (target != class_index)).sum().float()
        false_negative = ((prediction != class_index) & (target == class_index)).sum().float()
        f1_values.append((2 * true_positive + 1e-6) / (2 * true_positive + false_positive + false_negative + 1e-6))
    return {
        "accuracy": float(accuracy.detach().cpu()),
        "macro_f1": float(torch.stack(f1_values).mean().detach().cpu()),
        "samples": float(target.shape[0]),
    }


def classification_report(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_names: tuple[str, ...] = ("normal", "benign", "malignant"),
) -> dict[str, object]:
    """Return a JSON-friendly confusion matrix and per-class metrics."""

    prediction = logits.argmax(dim=1).long().cpu()
    target = target.long().cpu()
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, predicted_label in zip(target.tolist(), prediction.tolist()):
        if 0 <= true_label < num_classes and 0 <= predicted_label < num_classes:
            confusion[true_label, predicted_label] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for class_index, class_name in enumerate(class_names):
        true_positive = float(confusion[class_index, class_index])
        false_positive = float(confusion[:, class_index].sum() - confusion[class_index, class_index])
        false_negative = float(confusion[class_index, :].sum() - confusion[class_index, class_index])
        support = float(confusion[class_index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = int(confusion.sum())
    accuracy = float(torch.trace(confusion)) / total if total else 0.0
    return {
        "class_names": list(class_names),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "samples": total,
    }
