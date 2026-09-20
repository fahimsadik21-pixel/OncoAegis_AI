from __future__ import annotations

from typing import Any
import inspect

from app.core.specialist_registry import (
    get_or_create_specialist,
)

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


        specialist = get_or_create_specialist(
            model_id
        )


        execution_mode = None



        # -------------------------
        # Analysis based services
        # -------------------------

        if hasattr(specialist, "analyze"):

            execution_mode = "analyze"
            analyze_method = specialist.analyze
            parameters = inspect.signature(analyze_method).parameters
            kwargs = {}
            if "spacing_mm" in parameters:
                kwargs["spacing_mm"] = (
                    input_metadata.get("spacing_mm", (1.0, 1.0, 1.0))
                    if input_metadata
                    else (1.0, 1.0, 1.0)
                )
            if "input_metadata" in parameters:
                kwargs["input_metadata"] = input_metadata
            raw_output = analyze_method(input_data, **kwargs)



        # -------------------------
        # Prediction based services
        # -------------------------

        elif hasattr(specialist, "predict"):

            execution_mode = "predict"

            predict_method = specialist.predict


            parameters = inspect.signature(
                predict_method
            ).parameters


            kwargs = {}
            if "spacing_mm" in parameters:
                kwargs["spacing_mm"] = (
                    input_metadata.get("spacing_mm", (1.0, 1.0, 1.0))
                    if input_metadata
                    else (1.0, 1.0, 1.0)
                )
            if "input_metadata" in parameters:
                kwargs["input_metadata"] = input_metadata
            raw_output = predict_method(input_data, **kwargs)


        else:

            raise RuntimeError(
                f"No execution method found for {model_id}"
            )



        adapter = get_adapter(
            model_id
        )


        result = adapter.convert(
            raw_output,
            input_metadata=input_metadata,
        )


        # -------------------------
        # Execution provenance
        # -------------------------

        result.provenance.update(
            {
                "execution": {
                    "model_id": model_id,
                    "specialist_service":
                        type(specialist).__name__,
                    "adapter":
                        adapter.metadata(),
                    "mode":
                        execution_mode,
                }
            }
        )

        # Adapters may be shared by model aliases. The selected registry ID
        # is authoritative for this execution's provenance.
        if result.specialist is not None:
            result.specialist.model_id = model_id


        return result
