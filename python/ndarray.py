from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.dlpack import DLDeviceType


@dataclass
class NdArray:
    tensor: torch.Tensor

    def __dlpack__(self, stream: Any | None = None):
        return torch.utils.dlpack.to_dlpack(self.tensor)

    def __dlpack_device__(self):
        if self.tensor.device.type == "cpu":
            return (DLDeviceType.kDLCPU, 0)
        if self.tensor.device.type == "cuda":
            return (DLDeviceType.kDLCUDA, self.tensor.device.index or 0)
        raise RuntimeError(f"Unsupported device for DLPack: {self.tensor.device}")


def from_torch(tensor: torch.Tensor) -> NdArray:
    return NdArray(tensor=tensor)


def to_torch(array: NdArray) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(array)
