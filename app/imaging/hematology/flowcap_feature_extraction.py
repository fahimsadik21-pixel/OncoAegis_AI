"""
Patient-level FlowCAP AML feature extraction.

Each subject has 8 flow-cytometry tubes.

Each tube is summarized using robust statistical
features from the 7 measured channels.

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.data.hematology.flowcap_aml_loader import (
    EXPECTED_COLUMNS,
    NUM_SUBJECTS,
    load_flow_csv,
    subject_file_paths,
)


STAT_NAMES = [
    "mean",
    "std",
    "min",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "max",
]


def extract_tube_features(
    frame: pd.DataFrame,
):

    values = frame[
        EXPECTED_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    features = []

    feature_names = []

    event_count = float(
        values.shape[0]
    )

    features.append(
        np.log1p(
            event_count
        )
    )

    feature_names.append(
        "log_event_count"
    )

    for index, column in enumerate(
        EXPECTED_COLUMNS
    ):

        x = values[
            :,
            index,
        ]

        stats = [
            float(
                np.mean(x)
            ),

            float(
                np.std(x)
            ),

            float(
                np.min(x)
            ),

            float(
                np.quantile(
                    x,
                    0.10,
                )
            ),

            float(
                np.quantile(
                    x,
                    0.25,
                )
            ),

            float(
                np.median(x)
            ),

            float(
                np.quantile(
                    x,
                    0.75,
                )
            ),

            float(
                np.quantile(
                    x,
                    0.90,
                )
            ),

            float(
                np.max(x)
            ),
        ]

        features.extend(
            stats
        )

        for stat_name in STAT_NAMES:

            feature_names.append(
                f"{column}_{stat_name}"
            )

    return (
        np.asarray(
            features,
            dtype=np.float32,
        ),
        feature_names,
    )


def extract_subject_features(
    subject_id: int,
):

    paths = subject_file_paths(
        subject_id
    )

    patient_features = []

    patient_feature_names = []

    tube_event_counts = []

    for tube_index, path in enumerate(
        paths,
        start=1,
    ):

        frame = load_flow_csv(
            path
        )

        tube_event_counts.append(
            int(
                len(frame)
            )
        )

        tube_features, tube_names = (
            extract_tube_features(
                frame
            )
        )

        patient_features.extend(
            tube_features.tolist()
        )

        patient_feature_names.extend(
            [
                f"tube{tube_index}_{name}"
                for name
                in tube_names
            ]
        )

    return {

        "subject_id":
            int(
                subject_id
            ),

        "features":
            np.asarray(
                patient_features,
                dtype=np.float32,
            ),

        "feature_names":
            patient_feature_names,

        "tube_event_counts":
            tube_event_counts,

        "files": [
            path.name
            for path
            in paths
        ],
    }


def build_feature_cache(
    output_path: str | Path = (
        "datasets/processed/"
        "flowcap_aml/"
        "flowcap_patient_features.npz"
    ),
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_ids = []

    vectors = []

    feature_names = None

    for subject_id in range(
        1,
        NUM_SUBJECTS + 1,
    ):

        result = extract_subject_features(
            subject_id
        )

        subject_ids.append(
            subject_id
        )

        vectors.append(
            result[
                "features"
            ]
        )

        if feature_names is None:

            feature_names = result[
                "feature_names"
            ]

        print(
            f"Subject "
            f"{subject_id:03d}/"
            f"{NUM_SUBJECTS} "
            f"complete"
        )

    matrix = np.stack(
        vectors,
        axis=0,
    )

    np.savez_compressed(
        output_path,
        subject_ids=np.asarray(
            subject_ids,
            dtype=np.int32,
        ),
        features=matrix,
        feature_names=np.asarray(
            feature_names,
            dtype=str,
        ),
    )

    print(
        "CACHE COMPLETE"
    )

    print(
        "Shape:",
        matrix.shape
    )

    print(
        "Saved:",
        output_path
    )

    return output_path


if __name__ == "__main__":

    build_feature_cache()