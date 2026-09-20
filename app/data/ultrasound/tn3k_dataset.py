from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from app.data.ultrasound.tn3k_loader import (
    DEFAULT_ROOT,
    load_fold,
    load_labels,
)

from app.imaging.ultrasound.tn3k_preprocessing import (
    IMAGE_SIZE,
    preprocess_pair,
)


DatasetSplit = Literal[
    "train",
    "val",
    "test",
]


class TN3KDataset(Dataset):

    def __init__(
        self,
        split: DatasetSplit,
        *,
        fold: int = 0,
        root: Path = DEFAULT_ROOT,
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
    ):
        self.split = split
        self.fold = fold
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment

        if split in {"train", "val"}:

            labels = load_labels(
                "trainval",
                self.root,
            ).reset_index(drop=True)

            fold_data = load_fold(
                fold,
                self.root,
            )

            indices = fold_data[split]

            self.records = (
                labels.iloc[indices]
                .reset_index(drop=True)
            )

            self.image_dir = (
                self.root
                / "trainval-image"
            )

            self.mask_dir = (
                self.root
                / "trainval-mask"
            )

        elif split == "test":

            self.records = (
                load_labels(
                    "test",
                    self.root,
                )
                .reset_index(drop=True)
            )

            self.image_dir = (
                self.root
                / "test-image"
            )

            self.mask_dir = (
                self.root
                / "test-mask"
            )

        else:
            raise ValueError(
                "split must be train, val or test"
            )

        self._validate_records()


    def _validate_records(self) -> None:

        missing = []

        for row in self.records.itertuples(
            index=False
        ):
            filename = str(
                row.filename
            )

            image_path = (
                self.image_dir
                / filename
            )

            mask_path = (
                self.mask_dir
                / filename
            )

            if (
                not image_path.exists()
                or not mask_path.exists()
            ):
                missing.append(
                    filename
                )

        if missing:
            raise FileNotFoundError(
                "TN3K dataset contains missing "
                f"image/mask files: {missing[:20]}"
            )


    def __len__(self) -> int:
        return len(self.records)


    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # Horizontal flip
        if torch.rand(1).item() < 0.5:

            image = torch.flip(
                image,
                dims=[2],
            )

            mask = torch.flip(
                mask,
                dims=[2],
            )

        # Vertical flip
        if torch.rand(1).item() < 0.2:

            image = torch.flip(
                image,
                dims=[1],
            )

            mask = torch.flip(
                mask,
                dims=[1],
            )

        return image, mask


    def __getitem__(
        self,
        index: int,
    ):

        row = self.records.iloc[
            index
        ]

        filename = str(
            row["filename"]
        )

        label = int(
            row["label"]
        )

        image_path = (
            self.image_dir
            / filename
        )

        mask_path = (
            self.mask_dir
            / filename
        )

        image, mask = preprocess_pair(
            image_path,
            mask_path,
            image_size=self.image_size,
        )

        if (
            self.augment
            and self.split == "train"
        ):
            image, mask = self._augment(
                image,
                mask,
            )

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "filename": filename,
            "image_path": str(
                image_path
            ),
            "mask_path": str(
                mask_path
            ),
        }


def build_tn3k_datasets(
    fold: int = 0,
    image_size: int = IMAGE_SIZE,
):

    train_dataset = TN3KDataset(
        "train",
        fold=fold,
        image_size=image_size,
        augment=True,
    )

    val_dataset = TN3KDataset(
        "val",
        fold=fold,
        image_size=image_size,
        augment=False,
    )

    test_dataset = TN3KDataset(
        "test",
        image_size=image_size,
        augment=False,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":

    train_ds, val_ds, test_ds = (
        build_tn3k_datasets(
            fold=0
        )
    )

    print(
        "TRAIN:",
        len(train_ds),
    )

    print(
        "VAL:",
        len(val_ds),
    )

    print(
        "TEST:",
        len(test_ds),
    )

    sample = train_ds[0]

    print(
        "IMAGE:",
        sample["image"].shape,
        sample["image"].dtype,
        float(sample["image"].min()),
        float(sample["image"].max()),
    )

    print(
        "MASK:",
        sample["mask"].shape,
        sample["mask"].dtype,
        torch.unique(
            sample["mask"]
        ).tolist(),
    )

    print(
        "LABEL:",
        sample["label"].item(),
    )

    print(
        "FILE:",
        sample["filename"],
    )