"""Nodule-centred LUNA16 patch classification dataset."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.luna_annotations import annotation_voxel_centers, load_luna_annotations
from src.data.luna_dataset import _normalise_ct_slice, discover_luna16_cases


def _crop_with_padding(array: np.ndarray, center_x: int, center_y: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    left = center_x - half
    top = center_y - half
    right = left + patch_size
    bottom = top + patch_size
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - array.shape[1])
    pad_bottom = max(0, bottom - array.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        left += pad_left
        top += pad_top
    return array[top:bottom, left:right]


class LUNA16NoduleSliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Classify nodule-centred patches as annotated nodule or negative patch.

    LUNA16 provides nodule coordinates, not benign/malignant pathology. The
    positive examples are cropped around those coordinates so small nodules do
    not disappear when an entire 512x512 slice is resized.
    """

    def __init__(
        self,
        data_dir: str | Path,
        annotations_path: str | Path,
        raw_ct_dir: str | Path,
        *,
        case_ids: list[str] | tuple[str, ...] | set[str] | None = None,
        image_size: tuple[int, int] | None = (128, 128),
        patch_size: int = 160,
        augment: bool = False,
        negative_ratio: int = 1,
        seed: int = 42,
    ) -> None:
        if negative_ratio < 1:
            raise ValueError("negative_ratio must be at least 1")
        if patch_size < 16:
            raise ValueError("patch_size must be at least 16")
        all_cases = discover_luna16_cases(data_dir)
        selected = set(case_ids) if case_ids is not None else None
        self.cases = [case for case in all_cases if selected is None or case.case_id in selected]
        if not self.cases:
            raise ValueError("No LUNA16 cases selected")
        raw_root = Path(raw_ct_dir)
        annotations = load_luna_annotations(annotations_path, case_ids={case.case_id for case in self.cases})
        self.image_size = image_size
        self.patch_size = patch_size
        self.augment = augment
        self.slice_index: list[tuple[int, int, int, int, int]] = []
        rng = random.Random(seed)

        for case_index, case in enumerate(self.cases):
            processed_ct = np.load(case.ct_path, mmap_mode="r")
            processed_mask = np.load(case.mask_path, mmap_mode="r")
            raw_ct_path = raw_root / f"{case.case_id}.mhd"
            if not raw_ct_path.is_file():
                raise FileNotFoundError(f"Raw CT needed for annotation mapping not found: {raw_ct_path}")
            centers = annotation_voxel_centers(raw_ct_path, annotations.get(case.case_id, ()))
            centers = [center for center in centers if center[2] < processed_ct.shape[0]]
            positive_slices = {z for _x, _y, z, _diameter in centers}
            for x, y, z, _diameter in centers:
                self.slice_index.append((case_index, z, 1, x, y))

            negative_slices = [slice_id for slice_id in range(processed_ct.shape[0]) if slice_id not in positive_slices]
            desired_negative = min(
                len(negative_slices),
                max(len(centers) * negative_ratio, min(20, len(negative_slices))),
            )
            if desired_negative < len(negative_slices):
                local_rng = random.Random(rng.randint(0, 2**31 - 1) + case_index)
                negative_slices = local_rng.sample(negative_slices, desired_negative)
            for slice_id in negative_slices:
                mask_slice = np.asarray(processed_mask[slice_id]) > 0
                ys, xs = np.where(mask_slice)
                if len(xs):
                    point = rng.randrange(len(xs))
                    center_x, center_y = int(xs[point]), int(ys[point])
                else:
                    center_x = processed_ct.shape[2] // 2
                    center_y = processed_ct.shape[1] // 2
                self.slice_index.append((case_index, slice_id, 0, center_x, center_y))

        positive_entries = [entry for entry in self.slice_index if entry[2] == 1]
        negative_entries = [entry for entry in self.slice_index if entry[2] == 0]
        desired_negative = min(
            len(negative_entries),
            max(len(positive_entries) * negative_ratio, min(20, len(negative_entries))),
        )
        if desired_negative < len(negative_entries):
            negative_entries = rng.sample(negative_entries, desired_negative)
        self.slice_index = positive_entries + negative_entries
        if not self.slice_index:
            raise ValueError("No LUNA16 nodule patches selected")
        rng.shuffle(self.slice_index)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_index, slice_id, label, center_x, center_y = self.slice_index[index]
        case = self.cases[case_index]
        volume = np.load(case.ct_path, mmap_mode="r")
        source_slice = _normalise_ct_slice(volume[slice_id])
        patch = _crop_with_padding(source_slice, center_x, center_y, self.patch_size)
        image = torch.from_numpy(np.ascontiguousarray(patch)).float().unsqueeze(0)
        if self.image_size is not None:
            image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(-1,))
            if torch.rand(()) < 0.2:
                image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)
        return image, torch.tensor(label, dtype=torch.long)
