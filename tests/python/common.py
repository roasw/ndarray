from __future__ import annotations

import argparse
from pathlib import Path
import unittest

import torch


def parse_package_test_args(
    description: str, *, require_kernel_lib: bool
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--package-metadata", type=Path, required=True)
    parser.add_argument(
        "--kernel-lib", type=Path, required=require_kernel_lib, default=None
    )
    return parser.parse_args()


def parse_dual_package_test_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--package-metadata", type=Path, required=True)
    parser.add_argument("--kernel-package-metadata", type=Path, required=True)
    parser.add_argument("--kernel-lib", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    return parser.parse_args()


def parse_kernel_test_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--kernel-lib", type=Path, required=True)
    return parser.parse_args()


def load_packages(metadata_path: Path) -> dict[str, str]:
    package_paths: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            key, sep, value = line.partition("=")
            if not sep:
                raise RuntimeError(f"Invalid metadata line: {line}")
            if key in {"mode", "algorithm_module", "algorithm_class", "algorithm_name"}:
                continue
            package_paths[key] = value
    if not package_paths:
        raise RuntimeError(f"No package entries found in {metadata_path}")
    return package_paths


def factor_token(factor: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.ones(factor, dtype=dtype)


def run_test_case(test_case: type[unittest.TestCase]) -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1
