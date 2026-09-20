from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FLOWCAP_ROOT = PROJECT_ROOT / "datasets" / "hematology" / "flowcap_aml"

BASE_CSV_DIR = (
    FLOWCAP_ROOT / "CSV"
    if (FLOWCAP_ROOT / "CSV").exists()
    else FLOWCAP_ROOT / "csv"
)

JAR_PATH = FLOWCAP_ROOT / "Dream6-binary" / "Dream6C4.jar"
CLASSIFIER_PATH = (
    FLOWCAP_ROOT
    / "Dream6-binary"
    / "final-classifier.xml"
)

TARGET_SUBJECT_ID = 180
TUBES_PER_SUBJECT = 8

EXPECTED_COLUMNS = [
    "FS Lin",
    "SS Log",
    "FL1 Log",
    "FL2 Log",
    "FL3 Log",
    "FL4 Log",
    "FL5 Log",
]

DEFAULT_THRESHOLD = float(
    os.getenv("FLOWCAP_AML_THRESHOLD", "0.5")
)


class FlowCAPError(RuntimeError):
    pass


def _validate_environment() -> None:
    if not JAR_PATH.exists():
        raise FlowCAPError(
            f"Dream6C4.jar not found: {JAR_PATH}"
        )

    if not CLASSIFIER_PATH.exists():
        raise FlowCAPError(
            f"Classifier XML not found: {CLASSIFIER_PATH}"
        )

    if not BASE_CSV_DIR.exists():
        raise FlowCAPError(
            f"FlowCAP CSV directory not found: {BASE_CSV_DIR}"
        )

    files = list(BASE_CSV_DIR.glob("*.CSV"))

    if len(files) != 2872:
        raise FlowCAPError(
            f"Expected 2872 benchmark CSV files, "
            f"found {len(files)}"
        )


def validate_patient_csv(path: Path) -> dict:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise FlowCAPError(
            f"Could not read {path.name}: {exc}"
        ) from exc

    columns = list(df.columns)

    if columns != EXPECTED_COLUMNS:
        raise FlowCAPError(
            f"{path.name} has invalid columns.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found: {columns}"
        )

    if df.empty:
        raise FlowCAPError(
            f"{path.name} contains no events."
        )

    for column in EXPECTED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[EXPECTED_COLUMNS].isna().all(axis=None):
        raise FlowCAPError(
            f"{path.name} contains no usable numeric data."
        )

    return {
        "filename": path.name,
        "events": int(len(df)),
        "columns": columns,
    }


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        os.link(source, destination)
    except Exception:
        shutil.copy2(source, destination)


def _create_dummy_description(path: Path) -> None:
    """
    DREAM6 binary requires this file even in load=true mode.

    IMPORTANT:
    These are NOT ground-truth AML labels.

    The file only satisfies the original JAR's mandatory
    description-file dependency. Subjects 180-359 are the
    original challenge test subjects and their prediction
    does not use this dummy label field.
    """

    lines = ["Subject,Status"]

    for subject_id in range(1, 180):
        lines.append(
            f"{subject_id},Normal"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _benchmark_file_number(
    subject_id: int,
    tube_number: int,
) -> int:
    return (
        (subject_id - 1) * TUBES_PER_SUBJECT
        + tube_number
    )


def _prepare_sandbox(
    patient_files: Iterable[Path],
    sandbox_root: Path,
) -> None:

    csv_dir = sandbox_root / "csv"
    classifier_dir = sandbox_root / "Dream6-binary"

    csv_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Hard-link original benchmark files.
    for number in range(1, 2873):
        filename = f"{number:04d}.CSV"

        source = BASE_CSV_DIR / filename
        destination = csv_dir / filename

        _hardlink_or_copy(
            source,
            destination,
        )

    # Replace subject 180 with uploaded patient tubes.
    for tube_number, patient_file in enumerate(
        patient_files,
        start=1,
    ):
        number = _benchmark_file_number(
            TARGET_SUBJECT_ID,
            tube_number,
        )

        target = csv_dir / f"{number:04d}.CSV"

        if target.exists():
            target.unlink()

        shutil.copy2(
            patient_file,
            target,
        )

    _hardlink_or_copy(
        CLASSIFIER_PATH,
        classifier_dir / "final-classifier.xml",
    )

    _create_dummy_description(
        sandbox_root
        / "DREAM6_AML_TrainingSet.csv"
    )


def _parse_subject_output(
    stdout: str,
    subject_id: int,
) -> dict:

    pattern = re.compile(
        rf"^{subject_id}\s+(.+)$",
        re.MULTILINE,
    )

    match = pattern.search(stdout)

    if not match:
        raise FlowCAPError(
            f"Could not find patient {subject_id} "
            f"in DREAM6 output."
        )

    tokens = match.group(1).split()

    try:
        numeric = [
            float(value)
            for value in tokens
        ]
    except ValueError as exc:
        raise FlowCAPError(
            "Unexpected DREAM6 output format."
        ) from exc

    # Challenge-test rows:
    #
    # patient | test | 8 tube scores | combined score
    #
    if len(numeric) < 10:
        raise FlowCAPError(
            "DREAM6 output did not contain the expected "
            "tube and combined scores."
        )

    test_flag = int(numeric[0])

    scores = numeric[1:]

    tube_scores = scores[:8]
    combined_score = scores[8]

    return {
        "subject_id": subject_id,
        "test_flag": test_flag,
        "tube_scores": tube_scores,
        "combined_score": float(combined_score),
    }


def analyze_flowcap_patient(
    patient_files: list[Path],
    threshold: float = DEFAULT_THRESHOLD,
    threads: int = 4,
) -> dict:

    _validate_environment()

    if len(patient_files) != 8:
        raise FlowCAPError(
            "Exactly 8 flow-cytometry CSV files are required."
        )

    validation = [
        validate_patient_csv(path)
        for path in patient_files
    ]

    with tempfile.TemporaryDirectory(
        prefix="oncoaegis_flowcap_"
    ) as temporary_directory:

        sandbox_root = Path(
            temporary_directory
        )

        _prepare_sandbox(
            patient_files,
            sandbox_root,
        )

        command = [
            "java",
            "-jar",
            str(JAR_PATH),
            f"home={sandbox_root}",
            "desc=DREAM6_AML_TrainingSet.csv",
            "classifier=Dream6-binary/final-classifier.xml",
            "load=true",
            f"threads={threads}",
        ]

        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )

        if process.returncode != 0:
            tail = process.stdout[-5000:]

            raise FlowCAPError(
                "Official DREAM6 classifier failed.\n"
                f"{tail}"
            )

        parsed = _parse_subject_output(
            process.stdout,
            TARGET_SUBJECT_ID,
        )

    combined_score = parsed["combined_score"]

    research_prediction = (
        "AML-like"
        if combined_score >= threshold
        else "Non-AML-like"
    )

    return {
        "model": {
            "name": "DREAM6 / FlowCAP-II AML classifier",
            "source": "Jstacs DREAM6 C4",
            "model_type": "official_pretrained_classifier",
            "clinically_validated": False,
        },

        "input": {
            "type": "flow_cytometry",
            "tube_count": 8,
            "tubes": validation,
        },

        "prediction": {
            "research_prediction": research_prediction,
            "combined_score": combined_score,
            "threshold": threshold,
            "tube_scores": parsed["tube_scores"],
        },

        "safety": {
            "clinical_diagnosis": False,
            "aml_confirmation": False,
            "aml_exclusion": False,
            "model_clinically_validated": False,
            "hematopathology_review_required": True,
            "message": (
                "Research-only output. This model cannot "
                "confirm or exclude acute myeloid leukemia. "
                "Clinical diagnosis requires expert flow "
                "cytometry interpretation together with "
                "morphology, immunophenotyping, molecular/"
                "cytogenetic testing and clinical context."
            ),
        },
    }