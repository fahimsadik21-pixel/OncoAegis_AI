"""Inspect a local DICOM CT series from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.imaging.ct.pipeline import CTPipelineResult, run_ct_pipeline


def build_safe_summary(result: CTPipelineResult) -> dict[str, Any]:
    """Convert pipeline output into a JSON-safe, non-diagnostic summary."""

    preprocessed = result.preprocessed
    route = result.anatomy_route
    return {
        "status": "ct_volume_preprocessed",
        "modality": result.series.modality,
        "series_slices": len(result.series.slices),
        "volume_shape": list(result.series.shape),
        "body_region": result.series.safe_metadata.get("body_region"),
        "pixel_spacing_mm": list(result.series.pixel_spacing),
        "slice_spacing_mm": result.series.slice_spacing,
        "preprocessed_volume_shape": list(preprocessed.output_shape),
        "preprocessed_spacing_mm": list(preprocessed.output_spacing_mm),
        "hu_window": list(preprocessed.hu_window),
        "clipped_fraction": preprocessed.clipped_fraction,
        "candidate_organs": list(route.candidate_organs),
        "selected_organ": route.selected_organ,
        "anatomy_route_status": route.route_status,
        "volume_ready": True,
        "patient_identifiers_in_response": False,
        "warnings": result.series.warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect one local DICOM CT series safely"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a DICOM file or a directory containing one CT series",
    )
    args = parser.parse_args()

    try:
        result = run_ct_pipeline(args.path)
    except Exception as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(build_safe_summary(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

