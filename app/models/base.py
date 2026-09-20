from abc import ABC, abstractmethod
from typing import Any


class MedicalAIModel(ABC):

    model_name: str = "unknown"
    model_version: str = "0.0"

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        pass

    @abstractmethod
    def postprocess(self, prediction: Any) -> Any:
        pass

    def run(self, data: Any) -> Any:
        processed = self.preprocess(data)
        prediction = self.predict(processed)
        return self.postprocess(prediction)