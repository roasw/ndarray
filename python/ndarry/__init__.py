from __future__ import annotations

import importlib
import torch


def _bindings_module():
    return importlib.import_module("_ndarray")


def from_torch(tensor: torch.Tensor):
    return _bindings_module().from_torch(tensor)


def to_torch(array):
    return _bindings_module().to_torch(array)


class ndarray:
    def __new__(cls, shape, dtype: torch.dtype | None = None):
        bindings = _bindings_module()
        if dtype is None or dtype == torch.float32:
            return bindings.ndarray_f32(shape)
        if dtype == torch.float64:
            return bindings.ndarray_f64(shape)
        raise TypeError("ndarray dtype must be torch.float32 or torch.float64")

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


__all__ = ["ndarray", "ndarray_f32", "ndarray_f64", "from_torch", "to_torch"]
