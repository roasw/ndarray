#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Upsample2DFourier(nn.Module):
    def forward(self, x: torch.Tensor, factor_token: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError("Upsample expects a 2D tensor")
        if x.dtype != torch.float32:
            raise RuntimeError("Upsample expects float32 input")
        if factor_token.ndim != 1:
            raise RuntimeError("Upsample factor token must be a 1D tensor")
        if factor_token.dtype != torch.float32:
            raise RuntimeError("Upsample factor token must be float32")

        factor = factor_token.shape[0]
        h, w = x.shape
        spec = torch.fft.fft2(x)
        out = torch.fft.ifft2(spec, s=(h * factor, w * factor))
        return out.real.to(torch.float32) * (factor * factor)

    @classmethod
    def export(cls, **config: Any) -> dict[str, Any]:
        max_factor = int(config.get("max_factor", 8))
        if max_factor < 1:
            raise RuntimeError("max_factor must be >= 1")

        model = cls().eval()
        example_input = torch.randn(7, 9, dtype=torch.float32)
        example_factor_token = torch.ones(2, dtype=torch.float32)

        h_dim = torch.export.Dim("H", min=2)
        w_dim = torch.export.Dim("W", min=2)
        factor_dim = torch.export.Dim("F", min=1, max=max_factor)

        exported = torch.export.export(
            model,
            (example_input, example_factor_token),
            dynamic_shapes={
                "x": {0: h_dim, 1: w_dim},
                "factor_token": {0: factor_dim},
            },
        )

        model_name = "upsample_2d_fourier_model"
        return {model_name: exported}
