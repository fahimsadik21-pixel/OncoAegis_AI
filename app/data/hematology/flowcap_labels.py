from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LABEL_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "hematology"
    / "flowcap_aml"
    / "metadata"
    / "DREAM6_AML_TrainingSet.csv"
)


class FlowCAPLabelError(Exception):
    pass


def load_training_labels(
    path: str | Path = LABEL_PATH,
):

    path = Path(path)

    if not path.exists():
        raise FlowCAPLabelError(
            f"Training label file not found: {path}"
        )

    # Flexible parser:
    # supports either:
    # subject,label
    # or raw two-column CSV without headers.

    frame = pd.read_csv(
        path,
        header=None,
    )

    if frame.shape[1] < 2:
        raise FlowCAPLabelError(
            "Expected at least two columns: subject_id,label"
        )

    labels = {}

    for _, row in frame.iterrows():

        try:
            subject_id = int(row.iloc[0])
        except Exception:
            continue

        label_text = str(
            row.iloc[1]
        ).strip().lower()

        if "aml" in label_text:
            label = 1

        elif (
            "normal" in label_text
            or
            "healthy" in label_text
        ):
            label = 0

        else:
            continue

        labels[subject_id] = label

    if not labels:
        raise FlowCAPLabelError(
            "No usable AML/Normal labels found."
        )

    return labels


if __name__ == "__main__":

    labels = load_training_labels()

    aml = sum(
        1
        for value in labels.values()
        if value == 1
    )

    normal = sum(
        1
        for value in labels.values()
        if value == 0
    )

    print(
        {
            "subjects": len(labels),
            "AML": aml,
            "Normal": normal,
        }
    )