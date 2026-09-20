from __future__ import annotations

from typing import Any

from app.core.model_router import (
    route_specialist_model,
)

from app.core.specialist_registry import (
    get_or_create_specialist,
)

from app.adapters.registry import (
    get_adapter,
)

from app.core.execution_engine import (
    ExecutionEngine,
)


class AnalysisOrchestrator:
    """
    Universal OncoAegis execution coordinator.

    Responsibilities:
    - model routing
    - specialist loading
    - adapter selection
    - unified inference execution
    """


    def __init__(self):
        self.execution_engine = ExecutionEngine()



    def prepare_analysis(
        self,
        *,
        modality: str,
        organ: str | None = None,
        task: str | None = None,
        cancer_type: str | None = None,
    ) -> dict[str, Any]:


        route = route_specialist_model(
            modality=modality,
            organ=organ,
            task=task,
            cancer_type=cancer_type,
        )


        if route.selected_model_id is None:

            return {
                "status": "unavailable",
                "route": route.to_dict(),
            }



        model_id = route.selected_model_id


        specialist = get_or_create_specialist(
            model_id
        )


        adapter = get_adapter(
            model_id
        )


        return {

            "status": "ready",

            "model_id": model_id,

            "route": route.to_dict(),

            "specialist": type(
                specialist
            ).__name__,

            "adapter": adapter.metadata(),

        }



    def execute_analysis(
        self,
        *,
        model_id: str,
        input_data: Any,
        input_metadata: dict | None = None,
    ):

        """
        Execute selected specialist model
        and convert output into the universal
        StandardAnalysisResult format.
        """


        return self.execution_engine.execute(
            model_id=model_id,
            input_data=input_data,
            input_metadata=input_metadata,
        )