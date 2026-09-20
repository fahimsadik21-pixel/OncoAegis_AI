from abc import ABC, abstractmethod
from typing import Any


class BaseSpecialistAdapter(ABC):
    """
    Universal adapter contract.

    Every OncoAegis specialist adapter
    must follow this interface.
    """


    adapter_id: str = "unknown"


    @abstractmethod
    def convert(
        self,
        specialist_output: dict[str, Any]
    ):
        """
        Convert specialist-specific output
        into StandardAnalysisResult.
        """
        pass



    def validate(
        self,
        specialist_output: dict[str, Any]
    ) -> bool:
        """
        Basic validation hook.
        Child adapters may override.
        """

        return isinstance(
            specialist_output,
            dict
        )



    def metadata(self) -> dict[str, Any]:
        """
        Adapter information.
        """

        return {
            "adapter_id": self.adapter_id
        }