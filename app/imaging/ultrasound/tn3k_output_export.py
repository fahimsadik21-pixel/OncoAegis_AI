from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_tn3k_json(
    result: dict[str, Any],
    output_path: str | Path,
) -> Path:

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return output_path