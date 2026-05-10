#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import unittest

import torch

from algorithm.upsample_2d_fourier import Upsample2DFourier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CPU torch upsample kernel")
    parser.add_argument("--kernel-lib", type=Path, required=True)
    return parser.parse_args()


class KernelUpsample2DFourierTests(unittest.TestCase):
    eager_f32: Upsample2DFourier
    eager_f64: Upsample2DFourier

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        torch.ops.load_library(str(args.kernel_lib))
        cls.eager_f32 = Upsample2DFourier(input_dtype=torch.float32).eval()
        cls.eager_f64 = Upsample2DFourier(input_dtype=torch.float64).eval()

    def _assert_match(self, shape: tuple[int, int], factor: int, dtype: torch.dtype):
        py_input = torch.randn(*shape, dtype=dtype)
        py_factor_token = torch.ones(factor, dtype=torch.float32)

        eager = (
            self.__class__.eager_f32
            if dtype == torch.float32
            else self.__class__.eager_f64
        )
        py_expected = eager(py_input, py_factor_token)
        py_output = torch.ops.ndarray.upsample_2d_fourier_cpu(py_input, factor)

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
        with self.assertRaises(RuntimeError):
            _ = torch.ops.ndarray.upsample_2d_fourier_cpu(bad_input, 2)

    def test_invalid_factor_rejected(self):
        py_input = torch.randn(4, 6, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            _ = torch.ops.ndarray.upsample_2d_fourier_cpu(py_input, 0)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        KernelUpsample2DFourierTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
