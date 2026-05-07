#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class Upsample2xBilinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export bilinear upsample .pt2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = Upsample2xBilinear().eval()
    example_input = torch.randn(1, 1, 2, 2, dtype=torch.float32)

    exported = torch.export.export(model, (example_input,))
    torch._inductor.aoti_compile_and_package(exported, package_path=str(args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
