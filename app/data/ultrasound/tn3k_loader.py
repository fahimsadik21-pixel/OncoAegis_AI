from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "ultrasound"
    / "tn3k"
    / "datasets"
    / "tn3k"
)


SplitName = Literal["trainval", "test"]


def _image_dir(root: Path, split: SplitName) -> Path:
    return root / f"{split}-image"


def _mask_dir(root: Path, split: SplitName) -> Path:
    return root / f"{split}-mask"


def list_image_files(
    split: SplitName,
    root: Path = DEFAULT_ROOT,
) -> list[Path]:
    folder = _image_dir(root, split)

    if not folder.exists():
        raise FileNotFoundError(
            f"TN3K image directory not found: {folder}"
        )

    return sorted(folder.glob("*.jpg"))


def list_mask_files(
    split: SplitName,
    root: Path = DEFAULT_ROOT,
) -> list[Path]:
    folder = _mask_dir(root, split)

    if not folder.exists():
        raise FileNotFoundError(
            f"TN3K mask directory not found: {folder}"
        )

    return sorted(folder.glob("*.jpg"))


def get_pairs(
    split: SplitName,
    root: Path = DEFAULT_ROOT,
) -> list[tuple[Path, Path]]:
    images = list_image_files(split, root)
    masks = list_mask_files(split, root)

    image_map = {
        path.stem: path
        for path in images
    }

    mask_map = {
        path.stem: path
        for path in masks
    }

    image_ids = set(image_map)
    mask_ids = set(mask_map)

    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)

    if missing_masks or missing_images:
        raise RuntimeError(
            "TN3K image-mask pairing mismatch.\n"
            f"Missing masks: {missing_masks[:20]}\n"
            f"Missing images: {missing_images[:20]}"
        )

    return [
        (
            image_map[item_id],
            mask_map[item_id],
        )
        for item_id in sorted(image_ids)
    ]


def inspect_pair(
    image_path: Path,
    mask_path: Path,
) -> dict:
    with Image.open(image_path) as image:
        image_size = image.size
        image_mode = image.mode

    with Image.open(mask_path) as mask:
        mask_size = mask.size
        mask_mode = mask.mode

    return {
        "id": image_path.stem,
        "image": str(image_path),
        "mask": str(mask_path),
        "image_size": image_size,
        "mask_size": mask_size,
        "image_mode": image_mode,
        "mask_mode": mask_mode,
        "size_match": image_size == mask_size,
    }


def load_labels(
    split: SplitName,
    root: Path = DEFAULT_ROOT,
) -> pd.DataFrame:
    filename = (
        "label4trainval.csv"
        if split == "trainval"
        else "label4test.csv"
    )

    path = root / filename

    if not path.exists():
        raise FileNotFoundError(
            f"TN3K label file not found: {path}"
        )

    return pd.read_csv(
        path,
        header=None,
        names=["filename", "label"],
    )

def load_fold(
    fold: int,
    root: Path = DEFAULT_ROOT,
):
    if fold not in {0, 1, 2, 3, 4}:
        raise ValueError(
            "fold must be one of 0, 1, 2, 3, 4"
        )

    path = root / f"tn3k-trainval-fold{fold}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"TN3K fold file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def inspect_tn3k(
    root: Path = DEFAULT_ROOT,
) -> dict:
    train_pairs = get_pairs("trainval", root)
    test_pairs = get_pairs("test", root)

    train_bad_size = []
    test_bad_size = []

    for image_path, mask_path in train_pairs:
        info = inspect_pair(image_path, mask_path)

        if not info["size_match"]:
            train_bad_size.append(info["id"])

    for image_path, mask_path in test_pairs:
        info = inspect_pair(image_path, mask_path)

        if not info["size_match"]:
            test_bad_size.append(info["id"])

    train_labels = load_labels("trainval", root)
    test_labels = load_labels("test", root)

    return {
        "root": str(root),

        "trainval": {
            "images": len(list_image_files("trainval", root)),
            "masks": len(list_mask_files("trainval", root)),
            "pairs": len(train_pairs),
            "size_mismatches": len(train_bad_size),
        },

        "test": {
            "images": len(list_image_files("test", root)),
            "masks": len(list_mask_files("test", root)),
            "pairs": len(test_pairs),
            "size_mismatches": len(test_bad_size),
        },

        "labels": {
            "trainval_rows": len(train_labels),
            "test_rows": len(test_labels),
            "trainval_columns": list(train_labels.columns),
            "test_columns": list(test_labels.columns),
        },

        "folds": {
            str(fold): (
                root
                / f"tn3k-trainval-fold{fold}.json"
            ).exists()
            for fold in range(5)
        },
    }


if __name__ == "__main__":
    from pprint import pprint

    pprint(inspect_tn3k())