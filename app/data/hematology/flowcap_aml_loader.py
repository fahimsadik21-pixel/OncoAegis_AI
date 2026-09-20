"""
FlowCAP-II / DREAM6 AML patient-level loader.

359 subjects
8 flow-cytometry CSV tubes per subject
2872 CSV files total.

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "hematology"
    / "flowcap_aml"
)

CSV_ROOT = (
    DEFAULT_ROOT
    / "CSV"
)

EXPECTED_COLUMNS = [
    "FS Lin",
    "SS Log",
    "FL1 Log",
    "FL2 Log",
    "FL3 Log",
    "FL4 Log",
    "FL5 Log",
]

NUM_SUBJECTS = 359
TUBES_PER_SUBJECT = 8


class FlowCAPLoaderError(Exception):
    pass


def subject_file_numbers(
    subject_id: int,
) -> list[int]:

    if not 1 <= subject_id <= NUM_SUBJECTS:
        raise FlowCAPLoaderError(
            f"Invalid subject id: {subject_id}"
        )

    start = (
        (subject_id - 1)
        * TUBES_PER_SUBJECT
        + 1
    )

    return list(
        range(
            start,
            start + TUBES_PER_SUBJECT,
        )
    )


def subject_file_paths(
    subject_id: int,
    csv_root: str | Path = CSV_ROOT,
) -> list[Path]:

    csv_root = Path(
        csv_root
    )

    paths = []

    for file_number in subject_file_numbers(
        subject_id
    ):

        path = (
            csv_root
            /
            f"{file_number:04d}.CSV"
        )

        if not path.exists():

            raise FlowCAPLoaderError(
                f"Missing flow file: {path}"
            )

        paths.append(
            path
        )

    return paths


def load_flow_csv(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(
        path
    )

    frame = pd.read_csv(
        path
    )

    missing = [
        column
        for column
        in EXPECTED_COLUMNS
        if column not in frame.columns
    ]

    if missing:

        raise FlowCAPLoaderError(
            f"{path.name}: missing columns "
            f"{missing}"
        )

    frame = frame[
        EXPECTED_COLUMNS
    ].copy()

    for column in EXPECTED_COLUMNS:

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        axis=0,
        how="any",
    )

    if frame.empty:

        raise FlowCAPLoaderError(
            f"No valid events in {path}"
        )

    return frame


def inspect_flowcap_dataset():

    if not CSV_ROOT.exists():

        raise FlowCAPLoaderError(
            f"CSV directory not found: "
            f"{CSV_ROOT}"
        )

    csv_files = sorted(
        CSV_ROOT.glob("*.CSV")
    )

    first_subject_paths = (
        subject_file_paths(
            1
        )
    )

    first = load_flow_csv(
        first_subject_paths[0]
    )

    return {

        "csv_files":
            len(
                csv_files
            ),

        "subjects":
            len(csv_files)
            //
            TUBES_PER_SUBJECT,

        "tubes_per_subject":
            TUBES_PER_SUBJECT,

        "features":
            EXPECTED_COLUMNS,

        "subject_1_files": [
            path.name
            for path
            in first_subject_paths
        ],

        "first_file_events":
            len(
                first
            ),

        "first_file":
            first_subject_paths[
                0
            ].name,
    }


if __name__ == "__main__":

    print(
        inspect_flowcap_dataset()
    )