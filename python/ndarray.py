from __future__ import annotations

import torch

from _ndarray import (
    from_torch as _from_torch,
    ndarray_f32,
    ndarray_f64,
    to_torch as _to_torch,
)


def from_torch(tensor: torch.Tensor):
    return _from_torch(tensor)


def to_torch(array):
    return _to_torch(array)


class ndarray:
    def __new__(cls, shape, dtype: torch.dtype | None = None):
        if dtype is None or dtype == torch.float32:
            return ndarray_f32(shape)
        if dtype == torch.float64:
            return ndarray_f64(shape)
        raise TypeError("ndarray dtype must be torch.float32 or torch.float64")

    @staticmethod
    def from_torch(tensor: torch.Tensor):
        return from_torch(tensor)

    @staticmethod
    def to_torch(array):
        return to_torch(array)


__all__ = ["ndarray", "ndarray_f32", "ndarray_f64", "from_torch", "to_torch"]
