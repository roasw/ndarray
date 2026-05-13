#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import unittest

from common import parse_package_test_args, run_test_case


def parse_args() -> argparse.Namespace:
    return parse_package_test_args(
        "Validate AOT runtime input-check env behavior", require_kernel_lib=False
    )


class Upsample2DFourierRuntimeChecksTests(unittest.TestCase):
    package_metadata: str

    @classmethod
    def setUpClass(cls):
        args = parse_args()
        cls.package_metadata = str(args.package_metadata)

    def _run_case_subprocess(self, env_value: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["AOTI_RUNTIME_CHECK_INPUTS"] = env_value
        return subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "import torch\n"
                    "from pathlib import Path\n"
                    "meta = Path(sys.argv[1])\n"
                    "entries = {}\n"
                    "for line in meta.read_text().splitlines():\n"
                    "    if not line or line.startswith('mode='):\n"
                    "        continue\n"
                    "    key, value = line.split('=', 1)\n"
                    "    entries[key] = value\n"
                    "fn = torch._inductor.aoti_load_package(entries['cpu_f32'])\n"
                    "x = torch.randn(1, 3, dtype=torch.float32)\n"
                    "factor_token = torch.ones(2, dtype=torch.float32)\n"
                    "y = fn(x, factor_token)\n"
                    "print(tuple(y.shape))\n"
                ),
                self.__class__.package_metadata,
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def test_checks_enabled_rejects_small_shape(self):
        result = self._run_case_subprocess("1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dim value is too small", result.stderr)

    def test_checks_disabled_accepts_small_shape(self):
        result = self._run_case_subprocess("")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("(2, 6)", result.stdout)


def main() -> int:
    return run_test_case(Upsample2DFourierRuntimeChecksTests)


if __name__ == "__main__":
    raise SystemExit(main())
