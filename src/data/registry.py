"""Dataset registry for the modular OncoAegis data layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "name",
    "source",
    "modality",
    "organ_coverage",
    "cancer_lesion_purpose",
    "annotation_type",
    "expected_size",
    "license",
    "local_path",
    "download_status",
    "preprocessing_status",
    "intended_model",
)


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    source: str
    modality: tuple[str, ...]
    organ_coverage: tuple[str, ...]
    cancer_lesion_purpose: str
    annotation_type: tuple[str, ...]
    expected_size: str
    license: str
    local_path: str
    download_status: str
    preprocessing_status: str
    intended_model: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetEntry":
        missing = [field for field in REQUIRED_FIELDS if field not in value]
        if missing:
            raise ValueError(f"Dataset registry entry is missing: {missing}")
        return cls(
            name=str(value["name"]),
            source=str(value["source"]),
            modality=tuple(str(item) for item in value["modality"]),
            organ_coverage=tuple(str(item) for item in value["organ_coverage"]),
            cancer_lesion_purpose=str(value["cancer_lesion_purpose"]),
            annotation_type=tuple(str(item) for item in value["annotation_type"]),
            expected_size=str(value["expected_size"]),
            license=str(value["license"]),
            local_path=str(value["local_path"]),
            download_status=str(value["download_status"]),
            preprocessing_status=str(value["preprocessing_status"]),
            intended_model=str(value["intended_model"]),
        )


class DatasetRegistry:
    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.path = Path(registry_path) if registry_path else Path(__file__).resolve().parents[2] / "datasets" / "registry" / "datasets.json"
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset registry not found: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        values = payload.get("datasets")
        if not isinstance(values, list):
            raise ValueError("Dataset registry must contain a datasets list")
        self.entries = tuple(DatasetEntry.from_dict(value) for value in values)
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("Dataset registry names must be unique")

    def get(self, name: str) -> DatasetEntry:
        for entry in self.entries:
            if entry.name.lower() == name.lower():
                return entry
        raise KeyError(f"Dataset not registered: {name}")

    def list(self) -> tuple[DatasetEntry, ...]:
        return self.entries


def get_dataset_registry(registry_path: str | Path | None = None) -> DatasetRegistry:
    return DatasetRegistry(registry_path)
