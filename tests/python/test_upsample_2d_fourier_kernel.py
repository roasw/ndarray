#!/usr/bin/env python3

from __future__ import annotations

import argparse
import unittest
from typing import Callable

import torch
import torch.utils.dlpack

from common import (
    factor_token,
    load_packages,
    parse_package_test_args,
    run_test_case,
)
from ndarry import from_torch
from ndarry.algorithm.upsample_2d_fourier import Upsample2DFourier


def parse_args() -> argparse.Namespace:
    return parse_package_test_args(
        "Validate kernel-based upsample .pt2 package", require_kernel_lib=True
    )


class Upsample2DFourierKernelTests(unittest.TestCase):
    compiled_f32: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    compiled_f64: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    eager_f32: Upsample2DFourier
    eager_f64: Upsample2DFourier

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        torch.ops.load_library(str(args.kernel_lib))
        packages = load_packages(args.package_metadata)
        cls.compiled_f32 = torch._inductor.aoti_load_package(packages["cpu_f32"])
        cls.compiled_f64 = torch._inductor.aoti_load_package(packages["cpu_f64"])
        cls.eager_f32 = Upsample2DFourier().eval()
        cls.eager_f64 = Upsample2DFourier().eval()

    @staticmethod
    def _factor_token(factor: int) -> torch.Tensor:
        return factor_token(factor)

    def test_multiple_shapes_match_reference(self):
        shapes = [(3, 5), (4, 6), (7, 9), (8, 11), (15, 17)]
        for h, w in shapes:
            py_input = torch.randn(h, w, dtype=torch.float32)
            py_expected = self.__class__.eager_f32(py_input, self._factor_token(2))
            py_output = self.__class__.compiled_f32(py_input, self._factor_token(2))
            self.assertTrue(
                torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4)
            )

    def test_factor_4_matches_reference(self):
        py_input = torch.randn(5, 7, dtype=torch.float32)
        factor = 4
        py_expected = self.__class__.eager_f32(py_input, self._factor_token(factor))
        py_output = self.__class__.compiled_f32(py_input, self._factor_token(factor))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))

    def test_float64_package_matches_reference(self):
        py_input = torch.randn(5, 7, dtype=torch.float64)
        factor = 4
        py_expected = self.__class__.eager_f64(py_input, self._factor_token(factor))
        py_output = self.__class__.compiled_f64(py_input, self._factor_token(factor))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-10, rtol=1e-10))

    def test_matches_cpp_ndarray_input(self):
        py_input = torch.randn(3, 3, dtype=torch.float32)
        py_expected = self.__class__.eager_f32(py_input, self._factor_token(2))
        cpp_input = from_torch(py_input)
        py_output = self.__class__.compiled_f32(
            torch.utils.dlpack.from_dlpack(cpp_input), self._factor_token(2)
        )
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))


def main() -> int:
    return run_test_case(Upsample2DFourierKernelTests)


if __name__ == "__main__":
    raise SystemExit(main())
