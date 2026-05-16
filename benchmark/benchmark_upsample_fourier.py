#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from ndarry.algorithm.upsample_2d_fourier import Upsample2DFourier
from ndarry.op_names import UPSAMPLE_2D_FOURIER


WARMUP_ITERS = 3
TIMED_ITERS = 10


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


def factor_token(factor: int) -> torch.Tensor:
    return torch.ones(factor, dtype=torch.int64)


def measure(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    token: torch.Tensor,
) -> tuple[float, float]:
    for _ in range(WARMUP_ITERS):
        fn(x.clone(), token)

    times_ns: list[int] = []
    for _ in range(TIMED_ITERS):
        inp = x.clone()
        t0 = time.perf_counter_ns()
        _ = fn(inp, token)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    median_ns = statistics.median(times_ns)
    median_us = median_ns / 1_000

    out_h = x.shape[0] * token.shape[0]
    out_w = x.shape[1] * token.shape[0]
    elements = x.numel() + out_h * out_w
    throughput = elements * TIMED_ITERS / (sum(times_ns) / 1e9)

    return median_us, throughput


def run_benchmarks(
    reference_metadata_path: Path,
    kernel_metadata_path: Path,
    kernel_lib_path: Path,
):
    torch.ops.load_library(str(kernel_lib_path))

    ref_packages = load_packages(reference_metadata_path)
    kernel_packages = load_packages(kernel_metadata_path)

    eager = Upsample2DFourier().eval()
    aoti_runner = torch._inductor.aoti_load_package(ref_packages["cpu_f32"])
    kernel_op = getattr(torch.ops.ndarray, UPSAMPLE_2D_FOURIER)
    kernel_aoti_runner = torch._inductor.aoti_load_package(kernel_packages["cpu_f32"])

    paths: list[tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        ("eager", lambda x, t: eager(x, t)),
        ("aoti", aoti_runner),
        ("kernel", kernel_op),
        ("kernel-aoti", kernel_aoti_runner),
    ]

    # (7, 9, 2) uses the same input shape as the export example,
    # providing a baseline for the exact case the .pt2 was traced with.
    configs = [
        (7, 9, 2),
        (128, 128, 2),
        (256, 256, 2),
        (256, 256, 4),
        (512, 512, 2),
        (512, 512, 4),
    ]

    print("## 结果 (f32)")
    print()
    header = "| 形状 | 倍率 |" + "|".join(f" {name} (μs) " for name, _ in paths) + "|"
    sep = "|------|------|" + "|".join("-----------" for _ in paths) + "|"
    print(header)
    print(sep)

    for h, w, factor in configs:
        x = torch.randn(h, w, dtype=torch.float32)
        token = factor_token(factor)
        cells = [f"{h}×{w}", str(factor)]
        for name, fn in paths:
            median_us, _ = measure(fn, x, token)
            cells.append(f"{median_us:.1f}")
        print("| " + " | ".join(cells) + " |")

    print()
    print("## 结果 (f64)")
    print()
    print(header)
    print(sep)

    # Re-load f64 AOTI packages
    aoti_runner_f64 = torch._inductor.aoti_load_package(ref_packages["cpu_f64"])
    kernel_aoti_runner_f64 = torch._inductor.aoti_load_package(
        kernel_packages["cpu_f64"]
    )
    eager_f64 = Upsample2DFourier().eval()

    paths_f64: list[
        tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    ] = [
        ("eager", lambda x, t: eager_f64(x, t)),
        ("aoti", aoti_runner_f64),
        ("kernel", kernel_op),
        ("kernel-aoti", kernel_aoti_runner_f64),
    ]

    for h, w, factor in configs:
        x = torch.randn(h, w, dtype=torch.float64)
        token = factor_token(factor)
        cells = [f"{h}×{w}", str(factor)]
        for name, fn in paths_f64:
            median_us, _ = measure(fn, x, token)
            cells.append(f"{median_us:.1f}")
        print("| " + " | ".join(cells) + " |")


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
