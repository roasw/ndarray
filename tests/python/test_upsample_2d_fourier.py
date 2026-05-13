#!/usr/bin/env python3

from __future__ import annotations

from common import run_test_case
from upsample_2d_fourier_test_base import Upsample2DFourierTestBase


class Upsample2DFourierTests(Upsample2DFourierTestBase):
    package_description = "Validate upsample .pt2 package"
    require_kernel_lib = False


def main() -> int:
    return run_test_case(Upsample2DFourierTests)


if __name__ == "__main__":
    raise SystemExit(main())
