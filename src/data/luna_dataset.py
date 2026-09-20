"""Case and slice datasets for the processed LUNA16 volumes.

The split is deliberately case-based.  Slices from one CT volume must never
be spread across train and validation, otherwise validation scores are
overly-optimistic because neighbouring slices are nearly identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


TensorPairTransform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class LUNA16Case:
    """A paired processed CT volume and lung-mask volume."""

    case_id: str
    ct_path: Path
    mask_path: Path


def discover_luna16_cases(data_dir: str | Path, *, strict: bool = True) -> list[LUNA16Case]:
    """Discover paired ``*_ct.npy`` and ``*_mask.npy`` files.

    Parameters
    ----------
    data_dir:
        Directory containing the processed LUNA16 arrays.
    strict:
        Raise a clear error when a CT or mask has no matching pair.
    """

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"LUNA16 processed directory not found: {root}")

    ct_files = {path.name.removesuffix("_ct.npy"): path for path in root.glob("*_ct.npy")}
    mask_files = {path.name.removesuffix("_mask.npy"): path for path in root.glob("*_mask.npy")}
    ct_ids = set(ct_files)
    mask_ids = set(mask_files)
    missing_masks = sorted(ct_ids - mask_ids)
    missing_ct = sorted(mask_ids - ct_ids)

    if strict and (missing_masks or missing_ct):
        problems: list[str] = []
        if missing_masks:
            problems.append(f"missing masks for {missing_masks[:3]}")
        if missing_ct:
            problems.append(f"missing CT volumes for {missing_ct[:3]}")
        raise ValueError("LUNA16 CT/mask pairing failed: " + "; ".join(problems))

    return [
        LUNA16Case(case_id, ct_files[case_id], mask_files[case_id])
        for case_id in sorted(ct_ids & mask_ids)
    ]


def _normalise_ct_slice(ct_slice: np.ndarray) -> np.ndarray:
    """Return a float32 CT slice in the ``[0, 1]`` range.

    Processed arrays are already normalized.  The fallback HU conversion keeps
    the dataset robust when a caller points it at an unprocessed HU array.
    """

    values = np.asarray(ct_slice, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(finite.min()) >= -0.01 and float(finite.max()) <= 1.01:
        return np.clip(values, 0.0, 1.0)
    values = np.clip(values, -1000.0, 400.0)
    return ((values + 1000.0) / 1400.0).astype(np.float32, copy=False)


def _resize_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    image_size: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_size is None:
        return image, mask
    image = F.interpolate(image.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=image_size, mode="nearest").squeeze(0)
    return image, mask


class LUNA16SliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """On-demand 2D slices backed by processed 3D LUNA16 ``.npy`` files."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        case_ids: Iterable[str] | None = None,
        image_size: tuple[int, int] | None = (256, 256),
        augment: bool = False,
        transform: TensorPairTransform | None = None,
        include_empty: bool = True,
    ) -> None:
        cases = discover_luna16_cases(data_dir)
        selected = set(case_ids) if case_ids is not None else None
        self.cases = [case for case in cases if selected is None or case.case_id in selected]
        if not self.cases:
            raise ValueError("No LUNA16 cases selected")

        self.image_size = image_size
        self.augment = augment
        self.transform = transform
        self.slice_index: list[tuple[int, int]] = []

        for case_index, case in enumerate(self.cases):
            ct = np.load(case.ct_path, mmap_mode="r")
            mask = np.load(case.mask_path, mmap_mode="r")
            if ct.ndim != 3 or mask.ndim != 3:
                raise ValueError(f"Expected 3D arrays for {case.case_id}, got {ct.shape} and {mask.shape}")
            if ct.shape != mask.shape:
                raise ValueError(f"CT/mask shape mismatch for {case.case_id}: {ct.shape} != {mask.shape}")

            for slice_id in range(ct.shape[0]):
                if include_empty or np.any(mask[slice_id] > 0):
                    self.slice_index.append((case_index, slice_id))

        if not self.slice_index:
            raise ValueError("No LUNA16 slices selected")

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_index, slice_id = self.slice_index[index]
        case = self.cases[case_index]
        ct = np.load(case.ct_path, mmap_mode="r")
        mask = np.load(case.mask_path, mmap_mode="r")

        image_array = np.ascontiguousarray(_normalise_ct_slice(ct[slice_id]))
        mask_array = np.ascontiguousarray((mask[slice_id] > 0).astype(np.float32))
        image = torch.from_numpy(image_array).unsqueeze(0)
        target = torch.from_numpy(mask_array).unsqueeze(0)
        image, target = _resize_pair(image, target, self.image_size)

        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
            if torch.rand(()) < 0.15:
                image = torch.clamp(image * (0.9 + 0.2 * torch.rand(())), 0.0, 1.0)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image.float(), target.float()


# The name used by the earlier project checkpoint now points to the useful
# slice-level dataset while keeping old imports working.
LUNA16Dataset = LUNA16SliceDataset
