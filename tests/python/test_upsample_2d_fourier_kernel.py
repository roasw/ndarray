#!/usr/bin/env python3

from __future__ import annotations

from common import run_test_case
from upsample_2d_fourier_test_base import Upsample2DFourierTestBase


class Upsample2DFourierKernelTests(Upsample2DFourierTestBase):
    package_description = "Validate kernel-based upsample .pt2 package"
    require_kernel_lib = True


def main() -> int:
    return run_test_case(Upsample2DFourierKernelTests)


if __name__ == "__main__":
    raise SystemExit(main())
