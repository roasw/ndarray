#!/usr/bin/env python3

from __future__ import annotations

import argparse
import unittest
from typing import Any

import torch

from common import parse_kernel_test_args, run_test_case
from ndarry.algorithm.upsample_2d_fourier import Upsample2DFourier
from ndarry.op_names import UPSAMPLE_2D_FOURIER


def parse_args() -> argparse.Namespace:
    return parse_kernel_test_args("Validate CPU torch upsample kernel")


class KernelUpsample2DFourierTests(unittest.TestCase):
    eager_f32: Upsample2DFourier
    eager_f64: Upsample2DFourier

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        torch.ops.load_library(str(args.kernel_lib))
        cls.eager_f32 = Upsample2DFourier().eval()
        cls.eager_f64 = Upsample2DFourier().eval()

    @staticmethod
    def _kernel_op() -> Any:
        return getattr(torch.ops.ndarray, UPSAMPLE_2D_FOURIER)

    def _assert_match(self, shape: tuple[int, int], factor: int, dtype: torch.dtype):
        py_input = torch.randn(*shape, dtype=dtype)
        py_factor_token = torch.ones(factor, dtype=dtype)

        eager = (
            self.__class__.eager_f32
            if dtype == torch.float32
            else self.__class__.eager_f64
        )
        py_expected = eager(py_input, py_factor_token)
        op = self._kernel_op()
        py_output = op(py_input, py_factor_token)

        self.assertEqual(py_output.dtype, dtype)
        self.assertEqual(tuple(py_output.shape), (shape[0] * factor, shape[1] * factor))

        if dtype == torch.float32:
            self.assertTrue(
                torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4)
            )
        else:
            self.assertTrue(
                torch.allclose(py_output, py_expected, atol=1e-10, rtol=1e-10)
            )

    def test_float32_matches_reference(self):
        for shape in [(1, 1), (3, 5), (4, 6), (7, 9)]:
            for factor in (2, 4):
                self._assert_match(shape, factor, torch.float32)

    def test_float64_matches_reference(self):
        for shape in [(1, 1), (3, 5), (4, 6), (7, 9)]:
            for factor in (2, 4):
                self._assert_match(shape, factor, torch.float64)

    def test_invalid_rank_rejected(self):
        bad_input = torch.randn(1, 4, 6, dtype=torch.float32)
        op = self._kernel_op()
        with self.assertRaises(RuntimeError):
            _ = op(bad_input, torch.ones(2, dtype=torch.float32))

    def test_invalid_factor_rejected(self):
        py_input = torch.randn(4, 6, dtype=torch.float32)
        op = self._kernel_op()
        with self.assertRaises(RuntimeError):
            _ = op(py_input, torch.ones(0, dtype=torch.float32))


def main() -> int:
    return run_test_case(KernelUpsample2DFourierTests)


if __name__ == "__main__":
    raise SystemExit(main())
