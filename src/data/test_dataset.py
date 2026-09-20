r"""Small LUNA16 dataset smoke check.

Run from the project root with:
    .venv\Scripts\python.exe -m src.data.test_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.luna_dataset import LUNA16SliceDataset, discover_luna16_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/processed/luna16", type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    args = parser.parse_args()

    cases = discover_luna16_cases(args.data_dir)
    dataset = LUNA16SliceDataset(args.data_dir, image_size=(args.image_size, args.image_size))
    image, mask = dataset[0]

    print("CT cases:", len(cases))
    print("Total slices:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min(), image.max())
    print("Mask unique values:", torch.unique(mask))
    print("Dataset ready!")


if __name__ == "__main__":
    main()
