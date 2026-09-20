"""Deterministic, leakage-safe dataset splitting helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def split_ids(
    ids: Iterable[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split IDs into disjoint train/validation lists."""

    values = sorted(set(ids))
    if len(values) < 2:
        raise ValueError("At least two cases are required for a train/validation split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = random.Random(seed)
    rng.shuffle(values)
    val_count = max(1, min(len(values) - 1, round(len(values) * val_fraction)))
    val_ids = sorted(values[:val_count])
    train_ids = sorted(values[val_count:])
    return train_ids, val_ids


def stratified_split(
    items: Iterable[T],
    label_getter: Callable[[T], str | int],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[T], list[T]]:
    """Stratify a small labelled collection without external dependencies."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str | int, list[T]] = defaultdict(list)
    for item in items:
        groups[label_getter(item)].append(item)
    if not groups:
        raise ValueError("Cannot split an empty collection")

    rng = random.Random(seed)
    train: list[T] = []
    validation: list[T] = []
    for label in sorted(groups, key=str):
        group = list(groups[label])
        rng.shuffle(group)
        if len(group) == 1:
            val_count = 0
        else:
            val_count = max(1, round(len(group) * val_fraction))
            val_count = min(len(group) - 1, val_count)
        validation.extend(group[:val_count])
        train.extend(group[val_count:])

    if not train or not validation:
        raise ValueError("Stratified split needs at least one train and one validation sample")
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
