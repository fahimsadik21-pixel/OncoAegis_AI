"""Dataset adapters and deterministic data-loader factories."""

from src.data.busi_dataset import BUSIDataset, BUSISample, discover_busi_samples
from src.data.luna_dataset import LUNA16Case, LUNA16SliceDataset, discover_luna16_cases
from src.data.luna_nodule_dataset import LUNA16NoduleSliceDataset
from src.data.registry import DatasetEntry, DatasetRegistry, get_dataset_registry
from src.data.universal import StandardMedicalSample, UniversalDataset

__all__ = [
    "BUSIDataset",
    "BUSISample",
    "LUNA16Case",
    "LUNA16SliceDataset",
    "LUNA16NoduleSliceDataset",
    "DatasetEntry",
    "DatasetRegistry",
    "StandardMedicalSample",
    "UniversalDataset",
    "get_dataset_registry",
    "discover_busi_samples",
    "discover_luna16_cases",
]
