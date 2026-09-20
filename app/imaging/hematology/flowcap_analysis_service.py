"""Universal specialist wrapper for the official DREAM6 FlowCAP runner."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.imaging.hematology.flowcap_official_service import (
    DEFAULT_THRESHOLD,
    analyze_flowcap_patient,
)


class FlowCAPAnalysisService:
    """Adapt eight uploaded flow-cytometry tube CSVs to ``analyze``."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        threads: int = 4,
    ) -> None:
        self.threshold = float(threshold)
        self.threads = int(threads)

    def analyze(self, patient_files: Iterable[str | Path]) -> dict:
        paths = [Path(path) for path in patient_files]
        return analyze_flowcap_patient(
            paths,
            threshold=self.threshold,
            threads=self.threads,
        )
