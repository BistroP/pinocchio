"""Layer / token-position sweep utilities (cross-cutting).

Use these to pick the extraction layer/position: run direction extraction + the override check
across candidate layers (configs/models.yaml `layers`) and keep where the signal separates best.
"""
from __future__ import annotations

from typing import Callable, Sequence


def layer_sweep(fn: Callable[[int], object], layers: Sequence[int]) -> dict[int, object]:
    """fn(layer) -> metric. Returns {layer: metric}."""
    return {l: fn(l) for l in layers}


def position_sweep(fn: Callable[[int], object], positions: Sequence[int]) -> dict[int, object]:
    return {p: fn(p) for p in positions}


def grid_sweep(fn: Callable[[int, int], object], layers: Sequence[int],
               positions: Sequence[int]) -> dict[tuple[int, int], object]:
    return {(l, p): fn(l, p) for l in layers for p in positions}


def argbest(results: dict, key: Callable[[object], float], maximize: bool = True):
    """Pick the (layer|position|grid) key whose metric is best under `key`."""
    items = list(results.items())
    return (max if maximize else min)(items, key=lambda kv: key(kv[1]))[0]
