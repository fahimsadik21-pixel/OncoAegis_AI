from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hematology"
    / "C-NMC 2019"
)

CLASS_TO_INDEX = {
    "hem": 0,
    "all": 1,
}

INDEX_TO_CLASS = {
    0: "HEM",
    1: "ALL",
}


class CNMC2019LoaderError(Exception):
    pass


class CNMC2019Dataset(Dataset):

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_ROOT,
        folds: tuple[int, ...] | list[int] = (0, 1, 2),
    ):

        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise CNMC2019LoaderError(
                f"Dataset not found: {self.root_dir}"
            )

        self.samples = []

        for fold in folds:

            fold_root = self.root_dir / f"fold_{fold}"

            if not fold_root.exists():
                raise CNMC2019LoaderError(
                    f"Fold not found: {fold_root}"
                )

            for image_path in fold_root.rglob("*.bmp"):

                parent_name = image_path.parent.name.lower()

                if parent_name not in CLASS_TO_INDEX:
                    continue

                label = CLASS_TO_INDEX[parent_name]

                self.samples.append(
                    {
                        "path": image_path,
                        "label": label,
                        "fold": int(fold),
                    }
                )

        self.samples.sort(
            key=lambda x: str(x["path"]).lower()
        )

        if not self.samples:
            raise CNMC2019LoaderError(
                "No C-NMC 2019 BMP samples found."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["path"]

        image = Image.open(image_path).convert("RGB")

        image_np = np.asarray(
            image,
            dtype=np.uint8,
        ).copy()

        image_tensor = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .float()
        )

        return {
            "image": image_tensor,
            "label": int(sample["label"]),
            "class_name": INDEX_TO_CLASS[
                int(sample["label"])
            ],
            "fold": int(sample["fold"]),
            "case": image_path.stem,
            "image_path": str(image_path),
            "original_shape": tuple(
                int(v)
                for v in image_np.shape
            ),
        }


def inspect_cnmc2019():

    dataset = CNMC2019Dataset()

    counts = {
        "HEM": 0,
        "ALL": 0,
    }

    fold_counts = {
        0: {"HEM": 0, "ALL": 0},
        1: {"HEM": 0, "ALL": 0},
        2: {"HEM": 0, "ALL": 0},
    }

    for sample in dataset.samples:

        name = INDEX_TO_CLASS[
            int(sample["label"])
        ]

        counts[name] += 1

        fold_counts[
            int(sample["fold"])
        ][name] += 1

    first = dataset[0]

    return {
        "total": len(dataset),
        "class_counts": counts,
        "fold_counts": fold_counts,
        "first_case": first["case"],
        "first_shape": first["original_shape"],
        "first_label": first["class_name"],
    }


if __name__ == "__main__":
    print(inspect_cnmc2019())