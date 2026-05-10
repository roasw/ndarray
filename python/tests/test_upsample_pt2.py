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


class UpsamplePt2Tests(unittest.TestCase):
    compiled: Callable[[torch.Tensor], torch.Tensor]

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        cls.compiled = torch._inductor.aoti_load_package(str(args.package))

    def _make_inputs(self):
        py_input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        py_expected = torch.nn.functional.interpolate(
            py_input,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        return py_input, py_expected

    def test_pt2_matches_torch_input(self):
        py_input, py_expected = self._make_inputs()
        self.assertIsNotNone(self.__class__.compiled)
        py_output = self.__class__.compiled(py_input)
        self.assertTrue(torch.allclose(py_output, py_expected))

    def test_pt2_matches_cpp_ndarray_input(self):
        py_input, py_expected = self._make_inputs()
        self.assertIsNotNone(self.__class__.compiled)
        cpp_input = from_torch(py_input)
        py_output = self.__class__.compiled(torch.utils.dlpack.from_dlpack(cpp_input))
        self.assertTrue(torch.allclose(py_output, py_expected))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UpsamplePt2Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
