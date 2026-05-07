#!/usr/bin/env python3

from __future__ import annotations

import torch

from ndarray import from_torch, to_torch


def main() -> int:
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapped = from_torch(base)
    bridged = to_torch(wrapped)

    assert torch.equal(base, bridged)

    bridged[0, 0] = 123.0
    assert float(base[0, 0]) == 123.0

    base[1, 2] = 456.0
    assert float(bridged[1, 2]) == 456.0

    print("ndarray python DLPack bridge test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
