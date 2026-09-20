"""
Dataset validation utilities for OncoAegis AI.
"""

from __future__ import annotations

from pathlib import Path

from app.registry.dataset_registry import (
    dataset_registry,
    BASE_DIR,
)


class DatasetValidator:

    def __init__(self):
        self.registry = dataset_registry

    def validate_dataset(self, name: str):

        dataset = self.registry.get_by_name(name)

        if not dataset:
            return {
                "dataset": name,
                "status": "not_found"
            }

        checks = {}

        local_path = (
            BASE_DIR /
            dataset["local_path"]
        )

        checks["path_exists"] = local_path.exists()

        checks["download_status"] = (
            dataset.get(
                "download_status"
            )
        )

        checks["preprocessing_status"] = (
            dataset.get(
                "preprocessing_status"
            )
        )

        checks["model_assignment"] = bool(
            dataset.get(
                "intended_model"
            )
        )

        ready = (
            checks["path_exists"]
            and checks["model_assignment"]
        )

        return {
            "dataset": name,
            "ready": ready,
            "checks": checks
        }


    def validate_all(self):

        results = []

        for dataset in self.registry.all():

            results.append(
                self.validate_dataset(
                    dataset["name"]
                )
            )

        return results


dataset_validator = DatasetValidator()