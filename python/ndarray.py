from _ndarray import from_torch, ndarray, to_torch

ndarray.__torch_function__ = classmethod(ndarray.__torch_function__)

__all__ = ["ndarray", "from_torch", "to_torch"]
