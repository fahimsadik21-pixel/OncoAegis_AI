"""Evidence-first Phase 1 validation for registered OncoAegis specialists.

This module deliberately separates three states:

``completed``
    A held-out set with explicit labels was evaluated and metrics were
    produced.
``not_evaluable``
    The model can run, but the local data does not contain valid ground truth
    for the model's output.  No synthetic metric is generated.
``failed``
    The evaluator or data contract failed and needs engineering attention.

All metrics are research measurements.  They do not establish cancer
diagnosis, clinical utility, or external validity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from app.registry.model_registry import get_model_registry, get_model_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1_evaluation"
REPORT_SCHEMA = "oncoaegis.phase1.evaluation.v1"
SEED = 42


@dataclass(frozen=True)
class EvaluationSpec:
    model_id: str
    task: str
    split: str
    ground_truth: str
    evaluator: str | None
    limitations: tuple[str, ...] = ()


EVALUATION_SPECS: dict[str, EvaluationSpec] = {
    "luna16_lung_segmentation": EvaluationSpec(
        "luna16_lung_segmentation", "segmentation", "case-wise validation", "lung masks", "luna_segmentation",
        ("LUNA16 is an annotated research dataset, not pathology-confirmed cancer ground truth.",),
    ),
    "luna16_nodule_detector": EvaluationSpec(
        "luna16_nodule_detector", "lesion_detection", "case-wise validation", "annotated nodule labels", "luna_nodule",
        ("The target is an annotated nodule, not a pathology-confirmed malignancy label.",),
    ),
    "busi_breast_segmentation": EvaluationSpec(
        "busi_breast_segmentation", "segmentation", "deterministic stratified validation", "lesion masks", "busi",
        ("BUSI labels are dataset research categories and lesion masks; they do not confirm malignancy.", "BUSI has no reliable patient identifier for patient-wise grouping in the local layout."),
    ),
    "busi_breast_classifier": EvaluationSpec(
        "busi_breast_classifier", "classification", "deterministic stratified validation", "BUSI class labels", "busi",
        ("BUSI class labels are research categories and are not a clinical cancer diagnosis.", "BUSI has no reliable patient identifier for patient-wise grouping in the local layout."),
    ),
    "msd_brain_tumor_segmentation": EvaluationSpec(
        "msd_brain_tumor_segmentation", "segmentation", "case-wise validation", "MSD voxel labels", "brain",
        ("MSD labels are research segmentation labels; no independent clinical validation is implied.",),
    ),
    "msd_pancreas_segmentation": EvaluationSpec(
        "msd_pancreas_segmentation", "segmentation", "case-wise validation", "MSD voxel labels", "pancreas",
        ("The checkpoint produces pancreas and model-region labels; this is not a pathology-confirmed cancer endpoint.",),
    ),
    "msd_pancreas_tumor_segmentation": EvaluationSpec(
        "msd_pancreas_tumor_segmentation", "segmentation", "case-wise validation", "MSD voxel labels", "pancreas",
        ("The checkpoint produces pancreas and model-region labels; this is not a pathology-confirmed cancer endpoint.",),
    ),
    "msd_colon_tumor_segmentation": EvaluationSpec(
        "msd_colon_tumor_segmentation", "segmentation", "case-wise validation", "MSD voxel labels", "colon",
        ("MSD labels are research segmentation labels; no independent clinical validation is implied.",),
    ),
    "ircadb01_liver_tumor_segmentation": EvaluationSpec(
        "ircadb01_liver_tumor_segmentation", "segmentation", "case-wise validation", "IRCADb01 liver/tumor masks", "liver",
        ("IRCADb01 is a research dataset; imaging masks are not pathology confirmation.",),
    ),
    "isic2016_skin_lesion_segmentation": EvaluationSpec(
        "isic2016_skin_lesion_segmentation", "segmentation", "deterministic sample validation", "ISIC lesion masks", "skin",
        ("ISIC masks are research segmentation targets; no biopsy-confirmed malignancy endpoint is used here.",),
    ),
    "cnmc2019_all_cell_classifier": EvaluationSpec(
        "cnmc2019_all_cell_classifier", "classification", "official fold-2 validation", "C-NMC cell labels", "cnmc",
        ("C-NMC labels are cell-image research labels and do not establish patient-level leukemia diagnosis.",),
    ),
    "flowcap_aml_patient_classifier": EvaluationSpec(
        "flowcap_aml_patient_classifier", "classification", "not evaluable locally", "none suitable for model score", None,
        ("The local status CSV is not coupled to valid ground-truth labels for the official model output; it must not be used for accuracy.",),
    ),
    "tn3k_thyroid_nodule_segmentation": EvaluationSpec(
        "tn3k_thyroid_nodule_segmentation", "segmentation", "official test split", "TN3K nodule masks", "tn3k",
        ("TN3K masks are research segmentation labels; nodule segmentation does not confirm thyroid cancer.",),
    ),
}


def _json_number(value: float | int | np.number | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return _json_number(np.mean(values)) if values else None


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path, device: torch.device) -> Any:
    return torch.load(path, map_location=device, weights_only=False)


def _state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _select_validation_indices(ids: list[str], *, seed: int = SEED, fraction: float = 0.2) -> list[int]:
    if len(ids) < 2:
        raise ValueError("At least two cases are required for a validation split")
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    count = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
    return sorted(order[:count])


def _binary_case_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | bool]:
    prediction = prediction.bool().flatten()
    target = target.bool().flatten()
    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())
    tn = int((~prediction & ~target).sum().item())
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "dice": 1.0 if denominator == 0 else (2.0 * tp) / denominator,
        "iou": 1.0 if union == 0 else tp / union,
        "precision": 1.0 if tp + fp == 0 and fn == 0 else (tp / (tp + fp) if tp + fp else 0.0),
        "recall": 1.0 if tp + fn == 0 and fp == 0 else (tp / (tp + fn) if tp + fn else 0.0),
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
        "target_positive": bool(target.any().item()),
        "prediction_positive": bool(prediction.any().item()),
    }


def _segmentation_report(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(f"Segmentation shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
    rows = [_binary_case_metrics(prediction[index], target[index]) for index in range(prediction.shape[0])]
    positive_rows = [row for row in rows if row["target_positive"]]
    return {
        "sample_count": len(rows),
        "positive_sample_count": len(positive_rows),
        "empty_target_sample_count": len(rows) - len(positive_rows),
        "metrics": {
            key: _mean(float(row[key]) for row in rows)
            for key in ("dice", "iou", "precision", "recall", "specificity")
        },
        "positive_case_metrics": {
            key: _mean(float(row[key]) for row in positive_rows)
            for key in ("dice", "iou", "precision", "recall", "specificity")
        },
    }


def _multiclass_segmentation_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_names: dict[int, str],
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for class_id, class_name in class_names.items():
        per_class[class_name] = _segmentation_report(
            prediction == class_id,
            target == class_id,
        )
    return {
        "per_class": per_class,
        "foreground_union": _segmentation_report(prediction > 0, target > 0),
    }


def _binary_auroc(scores: list[float], targets: list[int]) -> float | None:
    positives = sum(int(target == 1) for target in targets)
    negatives = sum(int(target == 0) for target in targets)
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    positive_rank_sum = sum(ranks[index] for index, target in enumerate(targets) if target == 1)
    return _json_number((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _classification_report(logits: torch.Tensor, target: torch.Tensor, class_names: tuple[str, ...]) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=1).cpu()
    predictions = probabilities.argmax(dim=1).tolist()
    targets = target.long().cpu().tolist()
    count = len(targets)
    confusion = [[0 for _ in class_names] for _ in class_names]
    for truth, prediction in zip(targets, predictions):
        if 0 <= truth < len(class_names) and 0 <= prediction < len(class_names):
            confusion[truth][prediction] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for class_id, class_name in enumerate(class_names):
        tp = confusion[class_id][class_id]
        fp = sum(row[class_id] for row in confusion) - tp
        fn = sum(confusion[class_id]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[class_id])}
        f1_values.append(f1)
        recalls.append(recall)
    accuracy = sum(confusion[index][index] for index in range(len(class_names))) / count if count else 0.0
    report: dict[str, Any] = {
        "sample_count": count,
        "class_names": list(class_names),
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "macro_f1": _mean(f1_values),
        "balanced_accuracy": _mean(recalls),
        "per_class": per_class,
        "expected_calibration_error": _expected_calibration_error(probabilities, targets),
    }
    if len(class_names) == 2:
        report["auroc"] = _binary_auroc(
            [float(row[1]) for row in probabilities.tolist()],
            targets,
        )
    return report


def _expected_calibration_error(probabilities: torch.Tensor, targets: list[int], bins: int = 10) -> float | None:
    if not targets:
        return None
    confidence, prediction = probabilities.max(dim=1)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    total = len(targets)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += float(selected.float().mean()) * abs(
                float(confidence[selected].mean())
                - float((prediction[selected] == target_tensor[selected]).float().mean())
            )
    return _json_number(error)


def _base_report(model_id: str, device: torch.device, seed: int) -> dict[str, Any]:
    spec = EVALUATION_SPECS[model_id]
    model_spec = get_model_spec(model_id)
    if model_spec is None:
        raise KeyError(f"Model is not registered: {model_id}")
    checkpoint = PROJECT_ROOT / str(model_spec.checkpoint_path)
    report = {
        "schema": REPORT_SCHEMA,
        "model_id": model_id,
        "dataset": model_spec.dataset,
        "task": spec.task,
        "split": spec.split,
        "ground_truth": spec.ground_truth,
        "seed": seed,
        "device": str(device),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint) if checkpoint.is_file() else None,
        "research_only": True,
        "clinical_diagnosis": False,
        "limitations": list(spec.limitations),
    }
    return report


def _finish(report: dict[str, Any], *, status: str, metrics: dict[str, Any] | None = None, sample_count: int | None = None) -> dict[str, Any]:
    report["status"] = status
    report["sample_count"] = sample_count
    report["metrics"] = metrics or {}
    return report


def _evaluate_luna_segmentation(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from src.data.loaders import create_luna16_loaders
    from src.models.unet import UNet

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = UNet(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(_state_dict(checkpoint))
    image_size = int(checkpoint.get("metadata", {}).get("image_size", 256))
    report["input_size"] = [image_size, image_size]
    loaders = create_luna16_loaders(
        PROJECT_ROOT / "datasets" / "processed" / "luna16",
        batch_size=8,
        image_size=(image_size, image_size),
        seed=seed,
        pin_memory=device.type == "cuda",
        max_validation_slices=max_samples,
    )
    train_cases = set(loaders.train_case_ids)
    validation_cases = set(loaders.validation_case_ids)
    report["split_audit"] = {
        "train_count": len(train_cases),
        "validation_count": len(validation_cases),
        "overlap_count": len(train_cases & validation_cases),
        "leakage_free": not (train_cases & validation_cases),
        "split_source": "case-wise src.data.loaders split",
    }
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loaders.validation:
            predictions.append((torch.sigmoid(model(images.to(device))) >= 0.5).cpu())
            targets.append(labels.cpu() >= 0.5)
    return _finish(report, status="completed", metrics={"segmentation": _segmentation_report(torch.cat(predictions), torch.cat(targets))}, sample_count=sum(x.shape[0] for x in predictions))


def _evaluate_luna_nodule(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from src.data.loaders import create_luna_nodule_loaders
    from src.models.luna_models import LUNANoduleClassifier

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = LUNANoduleClassifier(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(_state_dict(checkpoint))
    loaders = create_luna_nodule_loaders(
        PROJECT_ROOT / "datasets" / "processed" / "luna16",
        PROJECT_ROOT / "datasets" / "universal_ct" / "luna16" / "annotations.csv",
        PROJECT_ROOT / "datasets" / "universal_ct" / "luna16" / "subset1",
        batch_size=32,
        image_size=(256, 256),
        patch_size=160,
        seed=seed,
        pin_memory=device.type == "cuda",
        max_validation_slices=max_samples,
    )
    train_cases = set(loaders.train_case_ids)
    validation_cases = set(loaders.validation_case_ids)
    report["split_audit"] = {
        "train_count": len(train_cases),
        "validation_count": len(validation_cases),
        "overlap_count": len(train_cases & validation_cases),
        "leakage_free": not (train_cases & validation_cases),
        "split_source": "case-wise src.data.loaders split",
    }
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loaders.validation:
            logits.append(model(images.to(device)).cpu())
            targets.append(labels.cpu())
    joined_logits = torch.cat(logits)
    joined_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"classification": _classification_report(joined_logits, joined_targets, ("no_annotated_nodule", "annotated_nodule"))}, sample_count=len(joined_targets))


def _evaluate_busi(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from src.data.loaders import create_busi_loaders
    from src.models.busi_models import BUSIMultiTaskUNet

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = BUSIMultiTaskUNet(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(_state_dict(checkpoint))
    loaders = create_busi_loaders(
        PROJECT_ROOT / "datasets" / "ultrasound" / "Dataset_BUSI" / "Dataset_BUSI_with_GT",
        batch_size=32,
        image_size=(256, 256),
        seed=seed,
        pin_memory=device.type == "cuda",
        max_validation_samples=max_samples,
    )
    train_ids = {sample.sample_id for sample in loaders.train_samples}
    validation_ids = {sample.sample_id for sample in loaders.validation_samples}
    report["split_audit"] = {
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "overlap_count": len(train_ids & validation_ids),
        "leakage_free": not (train_ids & validation_ids),
        "split_source": "deterministic stratified sample split; patient IDs unavailable",
    }
    mask_predictions: list[torch.Tensor] = []
    mask_targets: list[torch.Tensor] = []
    class_logits: list[torch.Tensor] = []
    class_targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loaders.validation:
            output = model(batch["image"].to(device))
            mask_predictions.append((torch.sigmoid(output["segmentation_logits"]) >= 0.5).cpu())
            mask_targets.append(batch["mask"].cpu() >= 0.5)
            class_logits.append(output["classification_logits"].cpu())
            class_targets.append(batch["label"].cpu())
    all_masks = torch.cat(mask_predictions)
    all_mask_targets = torch.cat(mask_targets)
    all_logits = torch.cat(class_logits)
    all_class_targets = torch.cat(class_targets)
    metrics = {
        "segmentation": _segmentation_report(all_masks, all_mask_targets),
        "classification": _classification_report(all_logits, all_class_targets, ("normal", "benign", "malignant")),
    }
    return _finish(report, status="completed", metrics=metrics, sample_count=len(all_class_targets))


def _evaluate_tn3k(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.data.ultrasound.tn3k_dataset import TN3KDataset
    from app.models.ultrasound.tn3k_unet import TN3KUNet

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    image_size = int(checkpoint.get("image_size", 256))
    model = TN3KUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = TN3KDataset("test", image_size=image_size, augment=False)
    report["split_audit"] = {
        "split_source": "official TN3K test split",
        "overlap_count": 0,
        "leakage_free": True,
    }
    if max_samples:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions.append((torch.sigmoid(model(batch["image"].to(device))) >= 0.5).cpu())
            targets.append(batch["mask"].cpu() >= 0.5)
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"segmentation": _segmentation_report(all_predictions, all_targets)}, sample_count=all_targets.shape[0])


def _evaluate_skin(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.models.dermatology.skin_lesion_unet import SkinLesionUNet2D
    from app.training.dermatology.train_skin_lesion import ProcessedSkinDataset

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = SkinLesionUNet2D().to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = ProcessedSkinDataset()
    validation_size = int(len(dataset) * 0.20)
    validation, split_audit = _torch_validation_subset(dataset, len(dataset) - validation_size, seed, max_samples)
    report["split_audit"] = split_audit
    loader = DataLoader(validation, batch_size=16, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions.append(model(batch["image"].to(device)).argmax(dim=1).cpu() == 1)
            targets.append(batch["mask"].cpu() == 1)
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"segmentation": _segmentation_report(all_predictions, all_targets)}, sample_count=all_targets.shape[0])


def _evaluate_cnmc(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.models.hematology.cnmc_classifier import CNMCClassifier
    from app.training.hematology.train_cnmc_classifier import ProcessedCNMCDataset

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = CNMCClassifier().to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = ProcessedCNMCDataset(folds=(2,), training=False)
    if max_samples:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits.append(model(batch["image"].to(device)).cpu())
            targets.append(batch["label"].cpu())
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"classification": _classification_report(all_logits, all_targets, ("HEM", "ALL"))}, sample_count=len(all_targets))


def _validation_subset(dataset: Any, ids: list[str], seed: int, max_samples: int | None) -> Subset:
    indices = _select_validation_indices(ids, seed=seed)
    if max_samples:
        indices = indices[:max_samples]
    return Subset(dataset, indices)


def _torch_validation_subset(
    dataset: Any,
    train_size: int,
    seed: int,
    max_samples: int | None,
) -> tuple[Subset, dict[str, Any]]:
    """Reproduce the trainer's torch.random_split without loading samples."""

    validation_size = len(dataset) - train_size
    if train_size < 1 or validation_size < 1:
        raise ValueError(f"Invalid train/validation sizes: {train_size}, {validation_size}")
    _, validation = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    indices = list(validation.indices)
    if max_samples:
        indices = indices[:max_samples]
    return Subset(dataset, indices), {
        "train_count": train_size,
        "validation_count": validation_size,
        "evaluated_validation_count": len(indices),
        "overlap_count": 0,
        "leakage_free": True,
        "split_source": "trainer-compatible torch.random_split",
    }


def _evaluate_brain(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.data.mri.msd_brain_loader import MSDBrainDataset
    from app.models.mri.brain_tumor_unet import BrainTumor3DUNet

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = BrainTumor3DUNet().to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = MSDBrainDataset(target_size=128)
    validation, split_audit = _torch_validation_subset(dataset, int(len(dataset) * 0.8), seed, max_samples)
    report["split_audit"] = split_audit
    loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions.append(model(batch["image"].to(device)).argmax(dim=1).cpu())
            targets.append(batch["mask"].cpu())
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"segmentation": _multiclass_segmentation_report(all_predictions, all_targets, {1: "tumor_class_1", 2: "tumor_class_2", 3: "tumor_class_3"})}, sample_count=all_targets.shape[0])


def _evaluate_colon(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.models.ct.colon_tumor_unet import ColonTumorUNet3D
    from app.training.ct.train_colon_tumor_fast import CachedColonDataset

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = ColonTumorUNet3D().to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = CachedColonDataset()
    validation_size = max(1, int(len(dataset) * 0.20))
    validation, split_audit = _torch_validation_subset(dataset, len(dataset) - validation_size, seed, max_samples)
    report["split_audit"] = split_audit
    loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions.append(model(batch["image"].float().to(device)).argmax(dim=1).cpu())
            targets.append(batch["label"].cpu())
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"segmentation": _multiclass_segmentation_report(all_predictions, all_targets, {1: "colon_tumor"})}, sample_count=all_targets.shape[0])


def _evaluate_pancreas(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.models.ct.pancreas_tumor_unet import PancreasTumor3DUNet
    from app.training.ct.train_pancreas_tumor_fast import CachedPancreasDataset

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = PancreasTumor3DUNet().to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = CachedPancreasDataset(preload=False)
    validation_size = max(1, int(len(dataset) * 0.20))
    validation, split_audit = _torch_validation_subset(dataset, len(dataset) - validation_size, seed, max_samples)
    report["split_audit"] = split_audit
    loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions.append(model(batch["image"].float().to(device)).argmax(dim=1).cpu())
            targets.append(batch["label"].cpu())
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"segmentation": _multiclass_segmentation_report(all_predictions, all_targets, {1: "pancreas", 2: "model_tumor_region"})}, sample_count=all_targets.shape[0])


def _evaluate_liver(report: dict[str, Any], device: torch.device, seed: int, max_samples: int | None) -> dict[str, Any]:
    from app.models.ct.liver_tumor_unet import LiverTumor3DUNet
    from app.training.ct.train_liver_tumor import LiverTumorROIDataset

    checkpoint = _load_checkpoint(PROJECT_ROOT / report["checkpoint"], device)
    model = LiverTumor3DUNet(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(_state_dict(checkpoint))
    dataset = LiverTumorROIDataset()
    train_size = max(1, int(len(dataset) * 0.8))
    if len(dataset) - train_size == 0:
        train_size -= 1
    validation, split_audit = _torch_validation_subset(dataset, train_size, seed, max_samples)
    report["split_audit"] = split_audit
    loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions.append(model(batch["image"].float().to(device)).argmax(dim=1).cpu())
            targets.append(batch["label"].cpu())
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    return _finish(report, status="completed", metrics={"segmentation": _multiclass_segmentation_report(all_predictions, all_targets, {1: "liver", 2: "liver_tumor"})}, sample_count=all_targets.shape[0])


_EVALUATORS: dict[str, Callable[..., dict[str, Any]]] = {
    "luna_segmentation": _evaluate_luna_segmentation,
    "luna_nodule": _evaluate_luna_nodule,
    "busi": _evaluate_busi,
    "tn3k": _evaluate_tn3k,
    "skin": _evaluate_skin,
    "cnmc": _evaluate_cnmc,
    "brain": _evaluate_brain,
    "colon": _evaluate_colon,
    "pancreas": _evaluate_pancreas,
    "liver": _evaluate_liver,
}


def evaluate_model(
    model_id: str,
    *,
    device: str | torch.device = "auto",
    seed: int = SEED,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate one registered model and return a JSON-safe report."""

    if model_id not in EVALUATION_SPECS:
        raise KeyError(f"No Phase 1 evaluation specification for {model_id}")
    if get_model_spec(model_id) is None:
        raise KeyError(f"Model is not registered: {model_id}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    report = _base_report(model_id, torch_device, seed)
    spec = EVALUATION_SPECS[model_id]
    if spec.evaluator is None:
        return _finish(
            report,
            status="not_evaluable",
            metrics={},
            sample_count=0,
        ) | {
            "ground_truth_available": False,
            "not_evaluable_reason": "No valid local ground truth is coupled to this official model output.",
        }
    evaluator = _EVALUATORS[spec.evaluator]
    try:
        return evaluator(report, torch_device, seed, max_samples)
    except Exception as exc:
        return _finish(
            report,
            status="failed",
            metrics={},
            sample_count=None,
        ) | {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def evaluate_all(
    *,
    device: str | torch.device = "auto",
    seed: int = SEED,
    max_samples: int | None = None,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Evaluate every registered model and write one report per model."""

    registered = [spec.model_id for spec in get_model_registry().list()]
    missing = sorted(set(registered) - set(EVALUATION_SPECS))
    if missing:
        raise RuntimeError(f"Registered models missing Phase 1 evaluation specs: {missing}")
    reports: dict[str, dict[str, Any]] = {}
    evaluator_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for model_id in registered:
        spec = EVALUATION_SPECS[model_id]
        model_spec = get_model_spec(model_id)
        cache_key = (spec.evaluator, model_spec.checkpoint_path if model_spec else None)
        if cache_key in evaluator_cache:
            report = copy.deepcopy(evaluator_cache[cache_key])
            report["model_id"] = model_id
            report["task"] = spec.task
            report["split"] = spec.split
            report["ground_truth"] = spec.ground_truth
            report["limitations"] = list(spec.limitations)
        else:
            report = evaluate_model(model_id, device=device, seed=seed, max_samples=max_samples)
            evaluator_cache[cache_key] = copy.deepcopy(report)
        reports[model_id] = report
    summary = {
        "schema": REPORT_SCHEMA,
        "phase": "Phase 1 - Model Evaluation & Validation",
        "seed": seed,
        "device": str(device),
        "registered_model_count": len(registered),
        "completed_count": sum(report["status"] == "completed" for report in reports.values()),
        "not_evaluable_count": sum(report["status"] == "not_evaluable" for report in reports.values()),
        "failed_count": sum(report["status"] == "failed" for report in reports.values()),
        "reports": reports,
        "clinical_claim": "none",
        "limitations": [
            "These are research-dataset measurements, not clinical validation.",
            "External, prospective, pathology-linked validation remains required.",
        ],
    }
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for model_id, report in reports.items():
            (output_path / f"{model_id}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (output_path / "phase1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-first Phase 1 model evaluation.")
    parser.add_argument("--model-id", action="append", dest="model_ids", help="Evaluate one model; repeat for multiple models.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional bounded smoke evaluation; omit for the full held-out split.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.model_ids:
        reports = {model_id: evaluate_model(model_id, device=args.device, seed=args.seed, max_samples=args.max_samples) for model_id in args.model_ids}
        result: Any = {"reports": reports}
    else:
        result = evaluate_all(device=args.device, seed=args.seed, max_samples=args.max_samples, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
