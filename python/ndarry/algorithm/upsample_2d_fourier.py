#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ALGORITHM_NAME = Path(__file__).stem


class Upsample2DFourier(nn.Module):
    @staticmethod
    def _validate_inputs(x: torch.Tensor, factor_token: torch.Tensor) -> None:
        if x.ndim != 2:
            raise RuntimeError("Upsample expects a 2D tensor")
        if factor_token.ndim != 1:
            raise RuntimeError("Upsample factor token must be a 1D tensor")
        if factor_token.dtype != x.dtype:
            raise RuntimeError("Upsample factor token dtype must match input dtype")

    @staticmethod
    def _factor_from_token(factor_token: torch.Tensor) -> int:
        return factor_token.shape[0]

    @staticmethod
    def _pad_spectrum(
        spec_centered: torch.Tensor, out_h: int, out_w: int
    ) -> torch.Tensor:
        h, w = spec_centered.shape
        pad_h = out_h - h
        pad_w = out_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(spec_centered, (pad_left, pad_right, pad_top, pad_bottom))

    @staticmethod
    def _ifft_and_scale(spec_out: torch.Tensor, factor: int) -> torch.Tensor:
        out = torch.fft.ifft2(spec_out)
        return out.real * (factor * factor)

    def forward(self, x: torch.Tensor, factor_token: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(x, factor_token)

        factor = self._factor_from_token(factor_token)
        h, w = x.shape
        out_h, out_w = h * factor, w * factor

        spec = torch.fft.fft2(x)
        spec_centered = torch.fft.fftshift(spec)
        spec_padded = self._pad_spectrum(spec_centered, out_h, out_w)
        spec_out = torch.fft.ifftshift(spec_padded)
        return self._ifft_and_scale(spec_out, factor)

    @classmethod
    def export(cls, **config: Any) -> dict[str, Any]:
        max_factor = int(config.get("max_factor", 8))
        if max_factor < 1:
            raise RuntimeError("max_factor must be >= 1")

        h_dim = torch.export.Dim("H", min=1)
        w_dim = torch.export.Dim("W", min=1)
        factor_dim = torch.export.Dim("F", min=1, max=max_factor)

        modules: dict[str, Any] = {}
        for suffix, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            model = cls().eval()
            example_input = torch.randn(7, 9, dtype=dtype)
            example_factor_token = torch.ones(2, dtype=dtype)
            exported = torch.export.export(
                model,
                (example_input, example_factor_token),
                dynamic_shapes={
                    "x": {0: h_dim, 1: w_dim},
                    "factor_token": {0: factor_dim},
                },
            )
            model_name = f"{ALGORITHM_NAME}_cpu_{suffix}_model"
            modules[model_name] = exported

        return modules
