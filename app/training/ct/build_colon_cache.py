"""
Build fast cached MSD Task10 Colon tensors.

The cache preserves full-volume context.

Research use only.
"""

from pathlib import Path

import torch

from app.data.ct.msd_colon_loader import MSDColonDataset
from app.imaging.ct.colon_preprocessing import preprocess_colon_volume


CACHE_DIR = Path(
    "datasets/processed/msd_colon_96"
)


def main():

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = MSDColonDataset()

    print(
        "Cases:",
        len(dataset)
    )

    print(
        "Cache dir:",
        CACHE_DIR
    )

    tumor_cases = 0

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]

        processed = preprocess_colon_volume(

            sample[
                "image"
            ]
            .squeeze(0)
            .numpy(),

            sample[
                "label"
            ]
            .numpy(),
        )

        contains_tumor = bool(
            torch.any(
                processed.label
                ==
                1
            )
        )

        if contains_tumor:
            tumor_cases += 1

        output_path = (
            CACHE_DIR
            /
            f"{sample['case']}.pt"
        )

        torch.save(
            {
                "image":
                    processed.image.half(),

                "label":
                    processed.label.to(
                        torch.uint8
                    ),

                "case":
                    sample["case"],

                "contains_tumor":
                    contains_tumor,

                "spacing_zyx_mm":
                    sample[
                        "spacing_zyx_mm"
                    ],

                "original_shape":
                    sample[
                        "tensor_shape_zyx"
                    ],
            },
            output_path,
        )

        print(
            f"[{index + 1}/{len(dataset)}] "
            f"{sample['case']} "
            f"tumor: {contains_tumor}"
        )

    print(
        "Tumor-containing cases:",
        tumor_cases
    )

    print(
        "COLON CACHE BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()