#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from common import factor_token, load_packages
from ndarry.algorithm.upsample_2d_fourier import Upsample2DFourier
from ndarry.op_names import UPSAMPLE_2D_FOURIER


WARMUP_ITERS = 3
TIMED_ITERS = 10

BENCHMARK_CONFIGS = [
    # (7, 9, 2) uses the same input shape as the export example,
    # providing a baseline for the exact case the .pt2 was traced with.
    (7, 9, 2),
    (128, 128, 2),
    (256, 256, 2),
    (256, 256, 4),
    (512, 512, 2),
    (512, 512, 4),
]


def measure(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    token: torch.Tensor,
) -> tuple[float, float]:
    for _ in range(WARMUP_ITERS):
        fn(x, token)

    times_ns: list[int] = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter_ns()
        fn(x, token)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    mean_ns = statistics.mean(times_ns)
    std_ns = statistics.stdev(times_ns) if len(times_ns) > 1 else 0.0

    return mean_ns / 1e6, std_ns / 1e6


def print_table(
    dtype_label: str,
    paths: list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]],
    configs: list[tuple[int, int, int]],
    dtype: torch.dtype,
):
    print(f"## 结果 ({dtype_label})")
    print()
    header = "| 形状 | 倍率 |" + "|".join(f" {name} (ms) " for name, _ in paths) + "|"
    sep = "|------|------|" + "|".join("-----------" for _ in paths) + "|"
    print(header)
    print(sep)

    for h, w, factor in configs:
        x = torch.randn(h, w, dtype=dtype)
        token = factor_token(factor)
        cells = [f"{h}×{w}", str(factor)]
        for _, fn in paths:
            mean_ms, std_ms = measure(fn, x, token)
            cells.append(f"{mean_ms:.2f}±{std_ms:.2f}")
        print("| " + " | ".join(cells) + " |")


def run_benchmarks(
    reference_metadata_path: Path,
    kernel_metadata_path: Path,
    kernel_lib_path: Path,
):
    torch.ops.load_library(str(kernel_lib_path))

    ref_packages = load_packages(reference_metadata_path)
    kernel_packages = load_packages(kernel_metadata_path)

    eager = Upsample2DFourier().eval()
    aoti_f32 = torch._inductor.aoti_load_package(ref_packages["cpu_f32"])
    aoti_f64 = torch._inductor.aoti_load_package(ref_packages["cpu_f64"])
    kernel_op = getattr(torch.ops.ndarray, UPSAMPLE_2D_FOURIER)
    kernel_aoti_f32 = torch._inductor.aoti_load_package(kernel_packages["cpu_f32"])
    kernel_aoti_f64 = torch._inductor.aoti_load_package(kernel_packages["cpu_f64"])

    paths_f32 = [
        ("eager", eager),
        ("aoti", aoti_f32),
        ("kernel", kernel_op),
        ("kernel-aoti", kernel_aoti_f32),
    ]

    paths_f64 = [
        ("eager", eager),
        ("aoti", aoti_f64),
        ("kernel", kernel_op),
        ("kernel-aoti", kernel_aoti_f64),
    ]

    print_table("f32", paths_f32, BENCHMARK_CONFIGS, torch.float32)
    print()
    print_table("f64", paths_f64, BENCHMARK_CONFIGS, torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Fourier upsampling")
    parser.add_argument("--package-metadata", type=Path, required=True)
    parser.add_argument("--kernel-package-metadata", type=Path, required=True)
    parser.add_argument("--kernel-lib", type=Path, required=True)
    args = parser.parse_args()

    run_benchmarks(
        args.package_metadata,
        args.kernel_package_metadata,
        args.kernel_lib,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
