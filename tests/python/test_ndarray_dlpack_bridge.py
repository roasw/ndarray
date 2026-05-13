#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import torch

from ndarry import from_torch, ndarray, new_f32, new_f64, to_torch


class NdarrayDlpackBridgeTestBase(unittest.TestCase):
    dtype: torch.dtype

    def _make_py_cpp_pair(self):
        py_base = torch.arange(6, dtype=self.dtype).reshape(2, 3)
        cpp_array = from_torch(py_base)
        return py_base, cpp_array

    def test_torch_roundtrip_equality(self):
        py_base, cpp_array = self._make_py_cpp_pair()
        py_bridged = to_torch(cpp_array)
        self.assertTrue(torch.equal(py_base, py_bridged))
        self.assertEqual(py_bridged.data_ptr(), cpp_array.data_ptr())

    def test_to_torch_pointer_identity(self):
        _, cpp_array = self._make_py_cpp_pair()
        py_bridged_a = to_torch(cpp_array)
        py_bridged_b = to_torch(cpp_array)

        self.assertEqual(py_bridged_a.data_ptr(), cpp_array.data_ptr())
        self.assertEqual(py_bridged_b.data_ptr(), cpp_array.data_ptr())

    def test_zero_copy_mutation_cpp_to_py(self):
        py_base, cpp_array = self._make_py_cpp_pair()
        py_bridged = to_torch(cpp_array)
        py_bridged[0, 0] = 123.0
        self.assertEqual(float(py_base[0, 0]), 123.0)

    def test_zero_copy_mutation_py_to_cpp(self):
        py_base, cpp_array = self._make_py_cpp_pair()
        py_bridged = to_torch(cpp_array)
        py_base[1, 2] = 456.0
        self.assertEqual(float(py_bridged[1, 2]), 456.0)

    def test_numpy_dlpack_shares_memory(self):
        py_base, cpp_array = self._make_py_cpp_pair()
        py_numpy_view = np.from_dlpack(cpp_array)

        self.assertFalse(py_numpy_view.flags.writeable)
        self.assertEqual(
            py_numpy_view.__array_interface__["data"][0], py_base.data_ptr()
        )

        py_base[1, 0] = 654.0
        self.assertEqual(float(py_numpy_view[1, 0]), 654.0)

    def test_torch_function_dispatch(self):
        py_base, cpp_array = self._make_py_cpp_pair()
        py_added = torch.add(cpp_array, cpp_array)
        self.assertTrue(torch.equal(py_added, py_base + py_base))

    def test_python_created_ndarray_roundtrip(self):
        py_base, _ = self._make_py_cpp_pair()
        cpp_created = ndarray([2, 3], dtype=self.dtype)
        py_created = to_torch(cpp_created)

        self.assertEqual(py_created.data_ptr(), cpp_created.data_ptr())

        py_created[:] = py_base

        py_roundtrip = to_torch(cpp_created)
        self.assertEqual(py_roundtrip.data_ptr(), cpp_created.data_ptr())
        self.assertTrue(torch.equal(py_roundtrip, py_base))

    def test_from_torch_pointer_identity(self):
        py_base, _ = self._make_py_cpp_pair()
        cpp_from_torch = ndarray.from_torch(py_base)
        self.assertEqual(cpp_from_torch.data_ptr(), py_base.data_ptr())

    def test_torch_like_helper_methods(self):
        py_base, _ = self._make_py_cpp_pair()
        cpp_from_torch = ndarray.from_torch(py_base)

        self.assertEqual(cpp_from_torch.dim(), py_base.dim())
        self.assertEqual(cpp_from_torch.ndim, py_base.dim())
        self.assertEqual(cpp_from_torch.size(), py_base.size())
        self.assertEqual(cpp_from_torch.size(0), py_base.size(0))
        self.assertEqual(cpp_from_torch.size(-1), py_base.size(-1))
        self.assertEqual(cpp_from_torch.stride(), py_base.stride())
        self.assertEqual(cpp_from_torch.stride(0), py_base.stride(0))
        self.assertEqual(cpp_from_torch.stride(-1), py_base.stride(-1))
        self.assertEqual(cpp_from_torch.numel(), py_base.numel())
        self.assertEqual(cpp_from_torch.dtype, py_base.dtype)
        self.assertEqual(cpp_from_torch.device, py_base.device)
        self.assertEqual(cpp_from_torch.is_contiguous(), py_base.is_contiguous())

        py_memory_format = torch.contiguous_format
        self.assertEqual(
            cpp_from_torch.is_contiguous(py_memory_format),
            py_base.is_contiguous(memory_format=py_memory_format),
        )

    def test_explicit_copy_ops_allocate_new_storage(self):
        py_base, cpp_array = self._make_py_cpp_pair()
        py_bridged = to_torch(cpp_array)

        py_added = torch.add(py_bridged, py_bridged)
        self.assertNotEqual(py_added.data_ptr(), py_bridged.data_ptr())

        cpp_clone = cpp_array.clone()
        self.assertNotEqual(cpp_clone.data_ptr(), cpp_array.data_ptr())

        py_clone = to_torch(cpp_clone)
        py_bridged[0, 0] = 999.0
        self.assertNotEqual(float(py_clone[0, 0]), float(py_bridged[0, 0]))

    def test_explicit_constructors_match_dtype_factory(self):
        cpp_f32 = ndarray([2, 3], dtype=torch.float32)
        cpp_f32_explicit = new_f32([2, 3])
        self.assertEqual(cpp_f32.dtype, torch.float32)
        self.assertEqual(cpp_f32_explicit.dtype, torch.float32)

        cpp_f64 = ndarray([2, 3], dtype=torch.float64)
        cpp_f64_explicit = new_f64([2, 3])
        self.assertEqual(cpp_f64.dtype, torch.float64)
        self.assertEqual(cpp_f64_explicit.dtype, torch.float64)

    def test_invalid_dtype_error_recommends_explicit_constructors(self):
        with self.assertRaises(TypeError) as err:
            _ = ndarray([2, 3], dtype=torch.int32)
        self.assertIn("new_f32()", str(err.exception))
        self.assertIn("new_f64()", str(err.exception))

    def test_repeated_roundtrip_keeps_aliasing(self):
        py_base, cpp_array = self._make_py_cpp_pair()

        for i in range(128):
            py_bridged = to_torch(cpp_array)
            self.assertEqual(py_bridged.data_ptr(), cpp_array.data_ptr())

            value = float(i + 1)
            py_bridged[0, 0] = value
            self.assertEqual(float(py_base[0, 0]), value)

            cpp_again = from_torch(py_bridged)
            self.assertEqual(cpp_again.data_ptr(), py_bridged.data_ptr())
            self.assertEqual(cpp_again.data_ptr(), cpp_array.data_ptr())

            py_roundtrip = to_torch(cpp_again)
            self.assertEqual(py_roundtrip.data_ptr(), py_base.data_ptr())
            self.assertEqual(float(py_roundtrip[0, 0]), value)

    def test_from_torch_survives_source_tensor_scope(self):
        def make_cpp_array():
            py_local = torch.arange(6, dtype=self.dtype).reshape(2, 3)
            cpp_local = from_torch(py_local)
            return cpp_local, py_local.data_ptr()

        cpp_array, source_ptr = make_cpp_array()
        self.assertEqual(cpp_array.data_ptr(), source_ptr)

        py_after = to_torch(cpp_array)
        self.assertEqual(py_after.data_ptr(), source_ptr)
        py_after[1, 2] = 321.0
        self.assertEqual(float(py_after[1, 2]), 321.0)

    def test_to_torch_survives_cpp_scope(self):
        def make_torch_view():
            cpp_local = ndarray([2, 3], dtype=self.dtype)
            py_local = to_torch(cpp_local)
            py_local[0, 1] = 77.0
            return py_local, cpp_local.data_ptr()

        py_view, cpp_ptr = make_torch_view()
        self.assertEqual(py_view.data_ptr(), cpp_ptr)
        self.assertEqual(float(py_view[0, 1]), 77.0)

        py_view[1, 0] = 88.0
        self.assertEqual(float(py_view[1, 0]), 88.0)

        cpp_again = from_torch(py_view)
        self.assertEqual(cpp_again.data_ptr(), py_view.data_ptr())


class NdarrayDlpackBridgeFloat32Tests(NdarrayDlpackBridgeTestBase):
    dtype = torch.float32


class NdarrayDlpackBridgeFloat64Tests(NdarrayDlpackBridgeTestBase):
    dtype = torch.float64


def main() -> int:
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            NdarrayDlpackBridgeFloat32Tests
        )
    )
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            NdarrayDlpackBridgeFloat64Tests
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
