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


_CONSTRUCTOR_SPECS = (
    ("new_f32", "ndarray_f32", "float32", torch.float32),
    ("new_f64", "ndarray_f64", "float64", torch.float64),
    ("new_i32", "ndarray_i32", "int32", torch.int32),
    ("new_i64", "ndarray_i64", "int64", torch.int64),
    ("new_c32", "ndarray_c32", "complex64", torch.complex64),
    ("new_c64", "ndarray_c64", "complex128", torch.complex128),
    ("new_b", "ndarray_b", "bool", torch.bool),
)

_BINDING_TYPE_NAMES = {binding_name for _, binding_name, _, _ in _CONSTRUCTOR_SPECS}


def _make_explicit_constructor(public_name: str, binding_name: str, dtype_name: str):
    def constructor(shape):
        return getattr(_bindings_module(), binding_name)(shape)

    constructor.__name__ = public_name
    constructor.__doc__ = f"Create an explicit {dtype_name} ndarray instance."
    return constructor


for _public_name, _binding_name, _dtype_name, _dtype in _CONSTRUCTOR_SPECS:
    globals()[_public_name] = _make_explicit_constructor(
        _public_name,
        _binding_name,
        _dtype_name,
    )

_DTYPE_TO_CONSTRUCTOR = {
    dtype: globals()[public_name] for public_name, _, _, dtype in _CONSTRUCTOR_SPECS
}


class ndarray:
    """Convenience factory that dispatches to typed ndarray bindings."""

    @overload
    def __new__(cls, shape, dtype: None = None) -> Any: ...

    @overload
    def __new__(cls, shape, dtype: torch.dtype) -> Any: ...

    def __new__(cls, shape, dtype: torch.dtype | None = None):
        if dtype is None:
            dtype = torch.float32

        constructor = _DTYPE_TO_CONSTRUCTOR.get(dtype)
        if constructor is not None:
            return constructor(shape)

        raise TypeError(
            "ndarray dtype must be one of torch.float32/float64/int32/int64/"
            "complex64/complex128/bool. Use new_f32/new_f64/new_i32/new_i64/"
            "new_c32/new_c64/new_b for explicit constructors."
        )

    @staticmethod
    def from_torch(tensor: torch.Tensor):
        return from_torch(tensor)

    @staticmethod
    def to_torch(array):
        return to_torch(array)


def __getattr__(name: str):
    if name in _BINDING_TYPE_NAMES:
        return getattr(_bindings_module(), name)
    raise AttributeError(f"module 'ndarry' has no attribute '{name}'")


__all__ = [
    "ndarray",
    "ndarray_f32",
    "ndarray_f64",
    "ndarray_i32",
    "ndarray_i64",
    "ndarray_c32",
    "ndarray_c64",
    "ndarray_b",
    "new_f32",
    "new_f64",
    "new_i32",
    "new_i64",
    "new_c32",
    "new_c64",
    "new_b",
    "from_torch",
    "to_torch",
]
