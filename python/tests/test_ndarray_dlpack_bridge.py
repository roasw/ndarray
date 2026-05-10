#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import torch

from ndarray import from_torch, to_torch, ndarray


class NdarrayDlpackBridgeTests(unittest.TestCase):
    def _make_py_cpp_pair(self):
        py_base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
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
        cpp_created = ndarray([2, 3])
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


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NdarrayDlpackBridgeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
