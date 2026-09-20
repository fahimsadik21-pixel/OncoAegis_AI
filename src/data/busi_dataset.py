"""BUSI breast-ultrasound dataset adapter.

BUSI stores images and one or more mask files in class directories.  When an
image has multiple masks, they are merged with a pixel-wise maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


BUSI_CLASSES = ("normal", "benign", "malignant")
BUSI_CLASS_TO_INDEX = {name: index for index, name in enumerate(BUSI_CLASSES)}


@dataclass(frozen=True)
class BUSISample:
    sample_id: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    class_name: str
    label: int


def discover_busi_samples(data_dir: str | Path, *, require_masks: bool = True) -> list[BUSISample]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"BUSI directory not found: {root}")

    samples: list[BUSISample] = []
    for class_name in BUSI_CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"BUSI class directory not found: {class_dir}")
        for image_path in sorted(class_dir.glob("*.png")):
            if "_mask" in image_path.stem:
                continue
            mask_paths = tuple(sorted(class_dir.glob(f"{image_path.stem}_mask*.png")))
            if require_masks and not mask_paths:
                raise ValueError(f"No BUSI mask found for {image_path}")
            sample_id = f"{class_name}/{image_path.stem}"
            samples.append(
                BUSISample(
                    sample_id=sample_id,
                    image_path=image_path,
                    mask_paths=mask_paths,
                    class_name=class_name,
                    label=BUSI_CLASS_TO_INDEX[class_name],
                )
            )
    if not samples:
        raise ValueError(f"No BUSI images found in {root}")
    return samples


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _load_merged_mask(paths: Iterable[Path], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=np.float32)
    for path in paths:
        mask = _load_image(path)
        if mask.shape != shape:
            raise ValueError(f"BUSI image/mask shape mismatch: {path} has {mask.shape}, expected {shape}")
        merged = np.maximum(merged, mask)
    return (merged > 0).astype(np.float32)


class BUSIDataset(Dataset[dict[str, object]]):
    """BUSI samples returned as image, binary lesion mask and class label."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        samples: Iterable[BUSISample] | None = None,
        image_size: tuple[int, int] | None = (256, 256),
        augment: bool = False,
    ) -> None:
        all_samples = list(samples) if samples is not None else discover_busi_samples(data_dir)
        if not all_samples:
            raise ValueError("No BUSI samples selected")
        self.samples = all_samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_array = _load_image(sample.image_path)
        mask_array = _load_merged_mask(sample.mask_paths, image_array.shape)
        image = torch.from_numpy(np.ascontiguousarray(image_array)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask_array)).unsqueeze(0)

        if self.image_size is not None:
            image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=self.image_size, mode="nearest").squeeze(0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(-1,))
                mask = torch.flip(mask, dims=(-1,))
            if torch.rand(()) < 0.2:
                image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)

        return {
            "image": image.float(),
            "mask": mask.float(),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "class_name": sample.class_name,
        }


BusiDataset = BUSIDataset
