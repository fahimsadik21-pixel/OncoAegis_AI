from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.data.ct.msd_pancreas_loader import MSDPancreasDataset


CACHE_DIR = Path("datasets/processed/msd_pancreas_96")

TARGET_SIZE = (96, 96, 96)

ROI_MARGIN = (12, 32, 32)


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(
        volume.astype(np.float32),
        -125.0,
        275.0,
    )

    volume = (
        volume + 125.0
    ) / 400.0

    return volume.astype(np.float32)


def get_bounds(label: np.ndarray):
    coords = np.argwhere(label > 0)

    if coords.size == 0:
        return (
            (0, label.shape[0]),
            (0, label.shape[1]),
            (0, label.shape[2]),
        )

    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0) + 1

    bounds = []

    for axis in range(3):
        start = max(
            0,
            int(minimum[axis] - ROI_MARGIN[axis]),
        )

        end = min(
            label.shape[axis],
            int(maximum[axis] + ROI_MARGIN[axis]),
        )

        bounds.append((start, end))

    return tuple(bounds)


def resize_image(image: np.ndarray):
    x = (
        torch.from_numpy(
            np.ascontiguousarray(image)
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    x = F.interpolate(
        x,
        size=TARGET_SIZE,
        mode="trilinear",
        align_corners=False,
    )

    return x.squeeze(0).squeeze(0)


def resize_label(label: np.ndarray):
    y = (
        torch.from_numpy(
            np.ascontiguousarray(label)
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    y = F.interpolate(
        y,
        size=TARGET_SIZE,
        mode="nearest",
    )

    return (
        y.squeeze(0)
        .squeeze(0)
        .to(torch.uint8)
    )


def main():
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = MSDPancreasDataset()

    print("Cases:", len(dataset))
    print("Cache dir:", CACHE_DIR)

    for index in range(len(dataset)):
        sample = dataset[index]

        image = (
            sample["image"]
            .squeeze(0)
            .numpy()
        )

        label = (
            sample["label"]
            .numpy()
        )

        image = normalize_ct(image)

        bounds = get_bounds(label)

        (z0, z1), (y0, y1), (x0, x1) = bounds

        image = image[
            z0:z1,
            y0:y1,
            x0:x1
        ]

        label = label[
            z0:z1,
            y0:y1,
            x0:x1
        ]

        image = resize_image(image)

        label = resize_label(label)

        contains_cancer = bool(
            torch.any(label == 2)
        )

        output = CACHE_DIR / f"{sample['case']}.pt"

        torch.save(
            {
                "image": image.half(),
                "label": label,
                "case": sample["case"],
                "contains_cancer": contains_cancer,
                "crop_bounds": bounds,
            },
            output,
        )

        print(
            f"[{index + 1}/{len(dataset)}]",
            sample["case"],
            "cancer:",
            contains_cancer,
        )

    print("CACHE BUILD COMPLETE")


if __name__ == "__main__":
    main()