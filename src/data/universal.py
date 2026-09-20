"""Common internal sample format for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class StandardMedicalSample:
    sample_id: str
    dataset_name: str
    modality: str
    organ: str | None
    image: torch.Tensor
    mask: torch.Tensor | None = None
    label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UniversalDataset(Dataset[StandardMedicalSample]):
    """Wrap a project dataset and expose one stable sample contract."""

    def __init__(
        self,
        source_dataset: Dataset,
        *,
        dataset_name: str,
        modality: str,
        organ: str | None = None,
    ) -> None:
        self.source_dataset = source_dataset
        self.dataset_name = dataset_name
        self.modality = modality
        self.organ = organ

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> StandardMedicalSample:
        item = self.source_dataset[index]
        if isinstance(item, dict):
            return StandardMedicalSample(
                sample_id=str(item.get("sample_id", index)),
                dataset_name=self.dataset_name,
                modality=self.modality,
                organ=self.organ,
                image=item["image"],
                mask=item.get("mask"),
                label=int(item["label"]) if item.get("label") is not None else None,
                metadata={"class_name": item.get("class_name")},
            )
        if isinstance(item, (tuple, list)) and len(item) == 2:
            image, target = item
            if self.dataset_name.lower() == "luna16":
                return StandardMedicalSample(
                    sample_id=str(index),
                    dataset_name=self.dataset_name,
                    modality=self.modality,
                    organ=self.organ,
                    image=image,
                    mask=target if self.modality == "CT" and getattr(target, "ndim", 0) == 3 else None,
                    label=int(target) if self.modality == "CT" and getattr(target, "ndim", 0) == 0 else None,
                )
            return StandardMedicalSample(
                sample_id=str(index),
                dataset_name=self.dataset_name,
                modality=self.modality,
                organ=self.organ,
                image=image,
                mask=target,
            )
        raise TypeError(f"Unsupported dataset item type: {type(item)!r}")
