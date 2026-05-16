from __future__ import annotations

import unittest
from typing import Callable

import torch
import torch.utils.dlpack

from common import factor_token, load_packages, parse_package_test_args
from ndarry import from_torch
from ndarry.algorithm.upsample_2d_fourier import Upsample2DFourier


class Upsample2DFourierTestBase(unittest.TestCase):
    package_description: str
    require_kernel_lib: bool

    compiled_f32: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    compiled_f64: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    eager_f32: Upsample2DFourier
    eager_f64: Upsample2DFourier

    @classmethod
    def setUpClass(cls):
        args = parse_package_test_args(
            cls.package_description, require_kernel_lib=cls.require_kernel_lib
        )
        if args.kernel_lib is not None:
            torch.ops.load_library(str(args.kernel_lib))
        packages = load_packages(args.package_metadata)
        cls.compiled_f32 = torch._inductor.aoti_load_package(packages["cpu_f32"])
        cls.compiled_f64 = torch._inductor.aoti_load_package(packages["cpu_f64"])
        cls.eager_f32 = Upsample2DFourier().eval()
        cls.eager_f64 = Upsample2DFourier().eval()

    def _make_inputs(self):
        py_input = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        )
        py_expected = self.__class__.eager_f32(py_input, self._factor_token(2))
        return py_input, py_expected

    @staticmethod
    def _factor_token(factor: int) -> torch.Tensor:
        return factor_token(factor)

    def test_multiple_shapes_match_reference(self):
        py_shapes = [(3, 5), (4, 6), (7, 9), (8, 11), (15, 17)]
        self.assertIsNotNone(self.__class__.compiled_f32)

        for h, w in py_shapes:
            py_input = torch.randn(h, w, dtype=torch.float32)
            py_expected = self.__class__.eager_f32(py_input, self._factor_token(2))
            py_output = self.__class__.compiled_f32(py_input, self._factor_token(2))
            self.assertTrue(
                torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4)
            )

    def test_invalid_rank_is_rejected(self):
        self.assertIsNotNone(self.__class__.compiled_f32)
        py_input = torch.randn(1, 4, 6, dtype=torch.float32)
        with self.assertRaises(Exception):
            output = self.__class__.compiled_f32(py_input, self._factor_token(2))
            _ = tuple(output.shape)

    def test_factor_4_matches_reference(self):
        self.assertIsNotNone(self.__class__.compiled_f32)
        py_input = torch.randn(5, 7, dtype=torch.float32)
        factor = 4
        py_expected = self.__class__.eager_f32(py_input, self._factor_token(factor))
        py_output = self.__class__.compiled_f32(py_input, self._factor_token(factor))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))

    def test_float64_package_matches_reference(self):
        self.assertIsNotNone(self.__class__.compiled_f64)
        py_input = torch.randn(5, 7, dtype=torch.float64)
        factor = 4
        py_expected = self.__class__.eager_f64(py_input, self._factor_token(factor))
        py_output = self.__class__.compiled_f64(py_input, self._factor_token(factor))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-10, rtol=1e-10))

    def test_matches_torch_input(self):
        py_input, py_expected = self._make_inputs()
        self.assertIsNotNone(self.__class__.compiled_f32)
        py_output = self.__class__.compiled_f32(py_input, self._factor_token(2))
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))

    def test_matches_cpp_ndarray_input(self):
        py_input, py_expected = self._make_inputs()
        self.assertIsNotNone(self.__class__.compiled_f32)
        cpp_input = from_torch(py_input)
        py_output = self.__class__.compiled_f32(
            torch.utils.dlpack.from_dlpack(cpp_input), self._factor_token(2)
        )
        self.assertTrue(torch.allclose(py_output, py_expected, atol=1e-4, rtol=1e-4))

    def test_input_unchanged(self):
        for shape in [(3, 5), (4, 6), (7, 9)]:
            for dtype in (torch.float32, torch.float64):
                x = torch.randn(*shape, dtype=dtype)
                token = self._factor_token(2)
                orig = x.clone()

                _ = self.__class__.eager_f32(x, token)
                self.assertTrue(
                    torch.equal(x, orig),
                    f"eager modified input ({shape}, {dtype})",
                )

                _ = self.__class__.compiled_f32(x, token)
                self.assertTrue(
                    torch.equal(x, orig),
                    f"compiled modified input ({shape}, {dtype})",
                )
