#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import unittest
from typing import Callable

import torch
import torch.utils.dlpack

from ndarray import from_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate upsample .pt2 package")
    parser.add_argument("--package", type=Path, required=True)
    return parser.parse_args()


def upsample_2d_fourier(input_tensor: torch.Tensor, factor: int = 2) -> torch.Tensor:
    h, w = input_tensor.shape
    spec = torch.fft.fft2(input_tensor)
    out = torch.fft.ifft2(spec, s=(h * factor, w * factor))
    return (out.real * input_tensor.new_tensor(float(factor * factor))).to(
        torch.float32
    )


class Upsample2DFourierTests(unittest.TestCase):
    compiled: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        cls.compiled = torch._inductor.aoti_load_package(str(args.package))

    def _make_inputs(self):
        py_input = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        )
        py_expected = upsample_2d_fourier(py_input)
        return py_input, py_expected

    @staticmethod
    def _factor_token(factor: int) -> torch.Tensor:
        return torch.ones(factor, dtype=torch.float32)

    def test_multiple_shapes_match_reference(self):
        py_shapes = [(3, 5), (4, 6), (7, 9), (8, 11), (15, 17)]
        self.assertIsNotNone(self.__class__.compiled)

        for h, w in py_shapes:
            py_input = torch.randn(h, w, dtype=torch.float32)
            py_expected = upsample_2d_fourier(py_input)
            py_output = self.__class__.compiled(py_input, self._factor_token(2))
            self.assertTrue(
                torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4)
            )

            py_sampled = py_output[::2, ::2]
            self.assertTrue(torch.allclose(py_sampled, py_input, atol=1e-4, rtol=1e-4))

    def test_invalid_rank_is_rejected(self):
        self.assertIsNotNone(self.__class__.compiled)
        py_input = torch.randn(1, 4, 6, dtype=torch.float32)
        with self.assertRaises(Exception):
            _ = self.__class__.compiled(py_input, self._factor_token(2))

    def test_factor_4_matches_reference(self):
        self.assertIsNotNone(self.__class__.compiled)
        py_input = torch.randn(5, 7, dtype=torch.float32)
        factor = 4
        py_expected = upsample_2d_fourier(py_input, factor=factor)
        py_output = self.__class__.compiled(py_input, self._factor_token(factor))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))

        py_sampled = py_output[::factor, ::factor]
        self.assertTrue(torch.allclose(py_sampled, py_input, atol=1e-4, rtol=1e-4))

    def test_matches_torch_input(self):
        py_input, py_expected = self._make_inputs()
        self.assertIsNotNone(self.__class__.compiled)
        py_output = self.__class__.compiled(py_input, self._factor_token(2))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))

    def test_matches_cpp_ndarray_input(self):
        py_input, py_expected = self._make_inputs()
        self.assertIsNotNone(self.__class__.compiled)
        cpp_input = from_torch(py_input)
        py_output = self.__class__.compiled(
            torch.utils.dlpack.from_dlpack(cpp_input), self._factor_token(2)
        )
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Upsample2DFourierTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
