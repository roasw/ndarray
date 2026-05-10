#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class Upsample2xFourier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError("Upsample expects a 2D tensor")
        if x.dtype != torch.float32:
            raise RuntimeError("Upsample expects float32 input")

        py_h, py_w = x.shape
        py_spec = torch.fft.fft2(x)
        py_out = torch.fft.ifft2(py_spec, s=(py_h * 2, py_w * 2))
        return (py_out.real * x.new_tensor(4.0)).to(torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Fourier upsample .pt2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = Upsample2xFourier().eval()
    example_input = torch.randn(7, 9, dtype=torch.float32)

    py_h = torch.export.Dim("H", min=2)
    py_w = torch.export.Dim("W", min=2)

    exported = torch.export.export(
        model,
        (example_input,),
        dynamic_shapes={"x": {0: py_h, 1: py_w}},
    )
    torch._inductor.aoti_compile_and_package(exported, package_path=str(args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
