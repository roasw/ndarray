#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ALGORITHM_NAME = Path(__file__).stem


class Upsample2DFourier(nn.Module):
    @staticmethod
    def _validate_inputs(x: torch.Tensor, factor_token: torch.Tensor) -> None:
        if x.ndim != 2:
            raise RuntimeError("Upsample expects a 2D tensor")
        if factor_token.ndim != 1:
            raise RuntimeError("Upsample factor token must be a 1D tensor")
        if factor_token.dtype != torch.int64:
            raise RuntimeError("Upsample factor token dtype must be int64")

    @staticmethod
    def _factor_from_token(factor_token: torch.Tensor) -> int:
        # torch.export requires all dynamic params as tensor inputs
        # (a plain int cannot carry dynamic-shape constraints).  We
        # encode the upsampling factor as the 1D tensor length;
        # the element values are never read -- only size(0) matters.
        return factor_token.shape[0]

    @staticmethod
    def _pad_spectrum(spec: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        # AOTI/inductor lacks c-shim implementations for complex tensor
        # slice_scatter / copy_ / full ops, falling back to proxy executor
        # which produces NaNs.  Avoid this by converting to real view
        # (last dim = 2 for real/imag), operating purely on float tensors,
        # then converting back to complex before IFFT.
        h, w = spec.shape
        spec_real = torch.view_as_real(spec)
        padded = torch.zeros(out_h, out_w, 2, dtype=spec_real.dtype, device=spec.device)

        h_pos = (h + 1) // 2
        w_pos = (w + 1) // 2

        padded[:h_pos, :w_pos, :] = spec_real[:h_pos, :w_pos, :]
        padded[out_h - h + h_pos :, :w_pos, :] = spec_real[h_pos:, :w_pos, :]
        padded[:h_pos, out_w - w + w_pos :, :] = spec_real[:h_pos, w_pos:, :]
        padded[out_h - h + h_pos :, out_w - w + w_pos :, :] = spec_real[
            h_pos:, w_pos:, :
        ]

        return torch.view_as_complex(padded)

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
        spec_padded = self._pad_spectrum(spec, out_h, out_w)
        return self._ifft_and_scale(spec_padded, factor)

    @classmethod
    def export(cls, **config: Any) -> dict[str, Any]:
        max_factor = int(config.get("max_factor", 8))
        if max_factor < 1:
            raise RuntimeError("max_factor must be >= 1")

        max_dim = int(config.get("max_dim", 4096))
        h_dim = torch.export.Dim("H", min=2, max=max_dim)
        w_dim = torch.export.Dim("W", min=2, max=max_dim)
        factor_dim = torch.export.Dim("F", min=1, max=max_factor)

        class ConstrainedModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x, factor_token):
                h, w = x.shape
                torch._check(h >= 2)
                torch._check(w >= 2)
                torch._check(h > (h + 1) // 2)
                torch._check(w > (w + 1) // 2)
                return self.base_model(x, factor_token)

        modules: dict[str, Any] = {}
        for suffix, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            base_model = cls().eval()
            model = ConstrainedModel(base_model)
            example_input = torch.randn(7, 9, dtype=dtype)
            example_factor_token = torch.ones(2, dtype=torch.int64)
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
