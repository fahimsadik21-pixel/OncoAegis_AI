"""Preprocess LUNA16 MetaImage volumes into model-ready NumPy arrays.

The CT is windowed and normalized, but it is *not* multiplied by the target
mask.  Applying the annotation to the input would leak the answer into the
segmentation model.  The saved mask is binary: background ``0`` and lung
``1``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_mhd(path: str | Path) -> tuple[np.ndarray, tuple[float, ...]]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), image.GetSpacing()


def normalize_ct(volume: np.ndarray, *, low_hu: float = -1000.0, high_hu: float = 400.0) -> np.ndarray:
    """Window HU values and return float32 values in ``[0, 1]``."""

    if high_hu <= low_hu:
        raise ValueError("high_hu must be greater than low_hu")
    volume = np.clip(np.asarray(volume, dtype=np.float32), low_hu, high_hu)
    return ((volume - low_hu) / (high_hu - low_hu)).astype(np.float32, copy=False)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any positive annotation label to a binary uint8 mask."""

    return (np.asarray(mask) > 0).astype(np.uint8)


def preprocess_case(ct_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    ct, _ = load_mhd(ct_path)
    mask, _ = load_mhd(mask_path)
    if ct.shape != mask.shape:
        raise ValueError(f"CT/mask shape mismatch: {ct.shape} != {mask.shape}")
    return normalize_ct(ct), binary_mask(mask)


def process_dataset(
    ct_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
) -> int:
    """Process all CT/mask pairs and return the number of saved cases."""

    ct_root = Path(ct_dir)
    mask_root = Path(mask_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    ct_files = sorted(ct_root.glob("*.mhd"))
    if not ct_files:
        raise FileNotFoundError(f"No .mhd CT files found in {ct_root}")

    saved = 0
    for index, ct_path in enumerate(ct_files, start=1):
        case_id = ct_path.stem
        mask_path = mask_root / f"{case_id}.mhd"
        if not mask_path.exists():
            print(f"Skipping {case_id}: matching mask not found")
            continue
        ct_output = output_root / f"{case_id}_ct.npy"
        mask_output = output_root / f"{case_id}_mask.npy"
        if not overwrite and ct_output.exists() and mask_output.exists():
            saved += 1
            continue

        ct, mask = preprocess_case(ct_path, mask_path)
        np.save(ct_output, ct)
        np.save(mask_output, mask)
        saved += 1
        print(f"Processed {saved}: {case_id} ({index}/{len(ct_files)})")

    print(f"Finished. Saved cases: {saved}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-dir", default="datasets/universal_ct/luna16/subset1", type=Path)
    parser.add_argument("--mask-dir", default="datasets/universal_ct/luna16/seg-lungs-LUNA16", type=Path)
    parser.add_argument("--output-dir", default="datasets/processed/luna16", type=Path)
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing processed pairs")
    args = parser.parse_args()
    process_dataset(args.ct_dir, args.mask_dir, args.output_dir, overwrite=not args.skip_existing)


if __name__ == "__main__":
    main()
