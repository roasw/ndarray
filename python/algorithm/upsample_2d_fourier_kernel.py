#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Upsample2DFourierKernel(nn.Module):
    def __init__(self, input_dtype: torch.dtype):
        super().__init__()
        self.input_dtype = input_dtype

    def forward(self, x: torch.Tensor, factor_token: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError("Upsample expects a 2D tensor")
        if x.dtype != self.input_dtype:
            raise RuntimeError("Upsample input has unexpected dtype")
        if factor_token.ndim != 1:
            raise RuntimeError("Upsample factor token must be a 1D tensor")
        if factor_token.dtype != torch.float32:
            raise RuntimeError("Upsample factor token must be float32")

        return torch.ops.ndarray.upsample_2d_fourier_cpu(x, factor_token)

    @classmethod
    def export(cls, **config: Any) -> dict[str, Any]:
        max_factor = int(config.get("max_factor", 8))
        if max_factor < 1:
            raise RuntimeError("max_factor must be >= 1")

        kernel_lib = config.get("kernel_lib")
        if not kernel_lib:
            raise RuntimeError("kernel_lib config is required")
        torch.ops.load_library(str(kernel_lib))

        h_dim = torch.export.Dim("H", min=1)
        w_dim = torch.export.Dim("W", min=1)
        factor_dim = torch.export.Dim("F", min=1, max=max_factor)

        modules: dict[str, Any] = {}
        for suffix, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            model = cls(input_dtype=dtype).eval()
            example_input = torch.randn(7, 9, dtype=dtype)
            example_factor_token = torch.ones(2, dtype=torch.float32)

            exported = torch.export.export(
                model,
                (example_input, example_factor_token),
                dynamic_shapes={
                    "x": {0: h_dim, 1: w_dim},
                    "factor_token": {0: factor_dim},
                },
            )
            model_name = f"upsample_2d_fourier_kernel_cpu_{suffix}_model"
            modules[model_name] = exported

        return modules
