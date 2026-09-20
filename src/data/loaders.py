"""Train/validation DataLoader factories for the supported datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.busi_dataset import BUSIDataset, BUSISample, discover_busi_samples
from src.data.luna_dataset import LUNA16SliceDataset, discover_luna16_cases
from src.data.luna_nodule_dataset import LUNA16NoduleSliceDataset
from src.data.splits import split_ids, stratified_split


@dataclass
class LUNA16Loaders:
    train: DataLoader
    validation: DataLoader
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]


@dataclass
class BUSILoaders:
    train: DataLoader
    validation: DataLoader
    train_samples: tuple[BUSISample, ...]
    validation_samples: tuple[BUSISample, ...]


@dataclass
class LUNANoduleLoaders:
    train: DataLoader
    validation: DataLoader
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _limit_dataset(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def create_luna16_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = 4,
    image_size: tuple[int, int] | None = (256, 256),
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_empty: bool = True,
    max_train_slices: int | None = None,
    max_validation_slices: int | None = None,
) -> LUNA16Loaders:
    cases = discover_luna16_cases(data_dir)
    train_ids, validation_ids = split_ids(
        (case.case_id for case in cases), val_fraction=val_fraction, seed=seed
    )
    train_dataset = LUNA16SliceDataset(
        data_dir,
        case_ids=train_ids,
        image_size=image_size,
        augment=True,
        include_empty=include_empty,
    )
    validation_dataset = LUNA16SliceDataset(
        data_dir,
        case_ids=validation_ids,
        image_size=image_size,
        augment=False,
        include_empty=include_empty,
    )
    train_dataset = _limit_dataset(train_dataset, max_train_slices)
    validation_dataset = _limit_dataset(validation_dataset, max_validation_slices)
    return LUNA16Loaders(
        train=_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed, pin_memory=pin_memory),
        validation=_loader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed, pin_memory=pin_memory),
        train_case_ids=tuple(train_ids),
        validation_case_ids=tuple(validation_ids),
    )


def create_busi_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = 8,
    image_size: tuple[int, int] | None = (256, 256),
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> BUSILoaders:
    samples = discover_busi_samples(data_dir)
    train_samples, validation_samples = stratified_split(
        samples, lambda sample: sample.class_name, val_fraction=val_fraction, seed=seed
    )
    train_dataset = BUSIDataset(data_dir, samples=train_samples, image_size=image_size, augment=True)
    validation_dataset = BUSIDataset(data_dir, samples=validation_samples, image_size=image_size, augment=False)
    train_dataset = _limit_dataset(train_dataset, max_train_samples)
    validation_dataset = _limit_dataset(validation_dataset, max_validation_samples)
    return BUSILoaders(
        train=_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed, pin_memory=pin_memory),
        validation=_loader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed, pin_memory=pin_memory),
        train_samples=tuple(train_samples),
        validation_samples=tuple(validation_samples),
    )


def create_luna_nodule_loaders(
    data_dir: str | Path,
    annotations_path: str | Path,
    raw_ct_dir: str | Path,
    *,
    batch_size: int = 8,
    image_size: tuple[int, int] | None = (256, 256),
    patch_size: int = 160,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    negative_ratio: int = 1,
    max_train_slices: int | None = None,
    max_validation_slices: int | None = None,
) -> LUNANoduleLoaders:
    cases = discover_luna16_cases(data_dir)
    train_ids, validation_ids = split_ids(
        (case.case_id for case in cases),
        val_fraction=val_fraction,
        seed=seed,
    )
    train_dataset = LUNA16NoduleSliceDataset(
        data_dir,
        annotations_path,
        raw_ct_dir,
        case_ids=train_ids,
        image_size=image_size,
        patch_size=patch_size,
        augment=True,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    validation_dataset = LUNA16NoduleSliceDataset(
        data_dir,
        annotations_path,
        raw_ct_dir,
        case_ids=validation_ids,
        image_size=image_size,
        patch_size=patch_size,
        augment=False,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    train_dataset = _limit_dataset(train_dataset, max_train_slices)
    validation_dataset = _limit_dataset(validation_dataset, max_validation_slices)
    return LUNANoduleLoaders(
        train=_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
        ),
        validation=_loader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
        ),
        train_case_ids=tuple(train_ids),
        validation_case_ids=tuple(validation_ids),
    )
