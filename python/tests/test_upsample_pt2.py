#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ndarray import from_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate upsample .pt2 package")
    parser.add_argument("--package", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compiled = torch._inductor.aoti_load_package(str(args.package))

    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    expected = torch.nn.functional.interpolate(
        x,
        scale_factor=2.0,
        mode="bilinear",
        align_corners=False,
    )

    out_torch = compiled(x)
    assert torch.allclose(out_torch, expected)

    wrapped = from_torch(x)
    out_wrapped = compiled(torch.utils.dlpack.from_dlpack(wrapped))
    assert torch.allclose(out_wrapped, expected)

    print("upsample pt2 test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
