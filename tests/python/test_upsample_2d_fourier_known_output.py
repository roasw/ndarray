#!/usr/bin/env python3

from __future__ import annotations

import argparse
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import (
    factor_token,
    load_packages,
    parse_dual_package_test_args,
    run_test_case,
)


def parse_args() -> argparse.Namespace:
    return parse_dual_package_test_args("Validate known upsample output and dump plots")


def reference_upsample_2d_fourier(x: torch.Tensor, factor: int) -> torch.Tensor:
    if x.ndim != 2:
        raise RuntimeError("reference expects 2D tensor")
    if factor < 1:
        raise RuntimeError("factor must be >= 1")

    h, w = x.shape
    out_h, out_w = h * factor, w * factor

    spec = torch.fft.fft2(x)
    spec_centered = torch.fft.fftshift(spec)

    pad_h = out_h - h
    pad_w = out_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    spec_padded = torch.nn.functional.pad(
        spec_centered, (pad_left, pad_right, pad_top, pad_bottom)
    )
    spec_out = torch.fft.ifftshift(spec_padded)
    out = torch.fft.ifft2(spec_out)
    return out.real * (factor * factor)


class Upsample2DFourierKnownOutputTests(unittest.TestCase):
    standard_f32: callable
    kernel_f32: callable
    plots_dir: str

    @classmethod
    def setUpClass(cls):
        args = parse_args()

        torch.ops.load_library(str(args.kernel_lib))

        std_packages = load_packages(args.package_metadata)
        kernel_packages = load_packages(args.kernel_package_metadata)

        cls.standard_f32 = torch._inductor.aoti_load_package(std_packages["cpu_f32"])
        cls.kernel_f32 = torch._inductor.aoti_load_package(kernel_packages["cpu_f32"])

        cls.plots_dir = args.plots_dir
        cls.plots_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _annotate_values(ax, data: torch.Tensor):
        values = data.detach().cpu().numpy()
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                ax.text(
                    c,
                    r,
                    f"{values[r, c]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    def _assert_case_and_plot(
        self, x: torch.Tensor, factor: int, plot_name: str, title_prefix: str
    ):
        token = factor_token(factor, dtype=torch.float32)

        expected = reference_upsample_2d_fourier(x, factor)
        out_standard = self.__class__.standard_f32(x, token)
        out_kernel = self.__class__.kernel_f32(x, token)

        self.assertTrue(torch.allclose(out_standard, expected, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(out_kernel, expected, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(out_standard, out_kernel, atol=1e-6, rtol=1e-6))

        plot_path = self.__class__.plots_dir / plot_name
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        axes[0].imshow(x.numpy(), cmap="viridis")
        axes[0].set_title(f"{title_prefix} Input")
        self._annotate_values(axes[0], x)
        axes[1].imshow(expected.numpy(), cmap="viridis")
        axes[1].set_title(f"{title_prefix} Reference")
        self._annotate_values(axes[1], expected)
        axes[2].imshow(out_standard.detach().cpu().numpy(), cmap="viridis")
        axes[2].set_title(f"{title_prefix} Standard AOT")
        self._annotate_values(axes[2], out_standard)
        axes[3].imshow(out_kernel.detach().cpu().numpy(), cmap="viridis")
        axes[3].set_title(f"{title_prefix} Kernel AOT")
        self._annotate_values(axes[3], out_kernel)
        for ax in axes:
            ax.axis("off")
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

        self.assertTrue(plot_path.exists())

    def test_known_output_and_plots(self):
        factor = 2
        odd_x = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self._assert_case_and_plot(
            odd_x, factor, "upsample_known_case_3x3_f32.png", "Odd 3x3"
        )

        even_x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        )
        self._assert_case_and_plot(
            even_x, factor, "upsample_known_case_2x2_f32.png", "Even 2x2"
        )


def main() -> int:
    return run_test_case(Upsample2DFourierKnownOutputTests)


if __name__ == "__main__":
    raise SystemExit(main())
