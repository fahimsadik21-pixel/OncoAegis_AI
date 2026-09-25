from __future__ import annotations

from typing import Any
import inspect

from app.core.specialist_registry import (
    get_or_create_specialist,
    release_specialist,
)

from app.deployment.resource_policy import model_eviction_enabled

from app.adapters.registry import (
    get_adapter,
)



class ExecutionEngine:
    """
    Universal OncoAegis execution layer.

    Handles:
    - specialist loading
    - predict/analyze execution
    - adapter conversion
    - execution provenance
    """



    def execute(
        self,
        *,
        model_id: str,
        input_data: Any,
        input_metadata: dict | None = None,
    ):
        """Run a specialist and convert its raw output to the shared contract."""

        specialist = None
        try:
            specialist = get_or_create_specialist(model_id)
            execution_mode = None

            if hasattr(specialist, "analyze"):
                execution_mode = "analyze"
                method = specialist.analyze
            elif hasattr(specialist, "predict"):
                execution_mode = "predict"
                method = specialist.predict
            else:
                raise RuntimeError(
                    f"No execution method found for {model_id}"
                )

            parameters = inspect.signature(method).parameters
            kwargs = {}
            if "spacing_mm" in parameters:
                kwargs["spacing_mm"] = (
                    input_metadata.get("spacing_mm", (1.0, 1.0, 1.0))
                    if input_metadata
                    else (1.0, 1.0, 1.0)
                )
            if "input_metadata" in parameters:
                kwargs["input_metadata"] = input_metadata
            raw_output = method(input_data, **kwargs)

            adapter = get_adapter(model_id)
            result = adapter.convert(
                raw_output,
                input_metadata=input_metadata,
            )
            result.provenance.update(
                {
                    "execution": {
                        "model_id": model_id,
                        "specialist_service": type(specialist).__name__,
                        "adapter": adapter.metadata(),
                        "mode": execution_mode,
                    }
                }
            )

            # Adapters may be shared by model aliases. The selected registry ID
            # is authoritative for this execution's provenance.
            if result.specialist is not None:
                result.specialist.model_id = model_id
            return result
        finally:
            # Small hosted instances cannot safely retain every 2D/3D model in
            # memory after an analysis. The policy is off on local workstations.
            if model_eviction_enabled():
                release_specialist(model_id)
