"""
Dataset registry for OncoAegis AI.

Loads available medical datasets and provides
filtering for modality, organ and task routing.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    BASE_DIR /
    "datasets" /
    "registry" /
    "datasets.json"
)


class DatasetRegistry:

    def __init__(self):
        self.datasets = self._load()

    def _load(self):
        if not DATASET_FILE.exists():
            raise FileNotFoundError(
                f"Dataset registry not found: {DATASET_FILE}"
            )

        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("datasets", [])

    def all(self):
        return self.datasets

    def get_by_name(self, name: str):
        for dataset in self.datasets:
            if dataset["name"].lower() == name.lower():
                return dataset

        return None

    def filter(
        self,
        modality: str | None = None,
        organ: str | None = None
    ):

        results = []

        for dataset in self.datasets:

            if modality:
                if modality.upper() not in [
                    m.upper()
                    for m in dataset.get("modality", [])
                ]:
                    continue

            if organ:
                if organ.lower() not in [
                    o.lower()
                    for o in dataset.get(
                        "organ_coverage",
                        []
                    )
                ]:
                    continue

            results.append(dataset)

        return results


dataset_registry = DatasetRegistry()