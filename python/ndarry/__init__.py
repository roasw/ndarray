from __future__ import annotations

import importlib
from typing import Any, overload

import torch


"""Public Python API for ndarray bindings.

`ndarray(...)` is a convenience factory, not a concrete runtime class. It
returns one of the bound extension types (`ndarray_f32` or `ndarray_f64`) based
on `dtype`.
"""


def _bindings_module():
    return importlib.import_module("_ndarray")


def from_torch(tensor: torch.Tensor):
    return _bindings_module().from_torch(tensor)


def to_torch(array):
    return _bindings_module().to_torch(array)


def new_f32(shape):
    """Create an explicit float32 ndarray instance."""
    return _bindings_module().ndarray_f32(shape)


def new_f64(shape):
    """Create an explicit float64 ndarray instance."""
    return _bindings_module().ndarray_f64(shape)


class ndarray:
    """Convenience factory that dispatches to typed ndarray bindings."""

    @overload
    def __new__(cls, shape, dtype: None = None) -> Any: ...

    @overload
    def __new__(cls, shape, dtype: torch.dtype) -> Any: ...

    def __new__(cls, shape, dtype: torch.dtype | None = None):
        if dtype is None or dtype == torch.float32:
            return new_f32(shape)
        if dtype == torch.float64:
            return new_f64(shape)
        raise TypeError(
            "ndarray dtype must be torch.float32 or torch.float64. "
            "Use new_f32() or new_f64() for explicit constructors."
        )

    @staticmethod
    def from_torch(tensor: torch.Tensor):
        return from_torch(tensor)

    @staticmethod
    def to_torch(array):
        return to_torch(array)


def __getattr__(name: str):
    if name in {"ndarray_f32", "ndarray_f64"}:
        return getattr(_bindings_module(), name)
    raise AttributeError(f"module 'ndarry' has no attribute '{name}'")


__all__ = [
    "ndarray",
    "ndarray_f32",
    "ndarray_f64",
    "new_f32",
    "new_f64",
    "from_torch",
    "to_torch",
]
