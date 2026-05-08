#!/usr/bin/env python3

from __future__ import annotations

import torch
import numpy as np

from ndarray import from_torch, to_torch, ndarray


def main() -> int:
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapped = from_torch(base)
    bridged = to_torch(wrapped)

    assert torch.equal(base, bridged)

    bridged[0, 0] = 123.0
    assert float(base[0, 0]) == 123.0

    base[1, 2] = 456.0
    assert float(bridged[1, 2]) == 456.0

    np_view = np.from_dlpack(wrapped)
    assert np_view.flags.writeable is False
    assert np_view.__array_interface__["data"][0] == base.data_ptr()

    base[1, 0] = 654.0
    assert float(np_view[1, 0]) == 654.0

    torch_added = torch.add(wrapped, wrapped)
    assert torch.equal(torch_added, base + base)

    py_created = ndarray([2, 3])
    py_created_t = to_torch(py_created)
    py_created_t[:] = base
    assert torch.equal(to_torch(py_created), base)

    py_created_from_torch = ndarray.from_torch(base)
    assert py_created_from_torch.data_ptr == base.data_ptr()

    print("ndarray python DLPack bridge test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
