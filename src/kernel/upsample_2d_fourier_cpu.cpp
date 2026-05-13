#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft2.h>
#include <ATen/ops/fft_fftshift.h>
#include <ATen/ops/fft_ifft2.h>
#include <ATen/ops/fft_ifftshift.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/real.h>
#include <torch/library.h>

namespace kernel {

namespace {

constexpr const char *kUpsampleOpName = "upsample_2d_fourier";

} // namespace

int64_t ValidateFactorToken(const at::Tensor &factor_token) {
    if (factor_token.dim() != 1) {
        throw std::runtime_error("upsample_2d_fourier expects 1D factor_token");
    }
    if (factor_token.scalar_type() != at::kFloat) {
        throw std::runtime_error(
            "upsample_2d_fourier expects float32 factor_token");
    }

    const int64_t factor = factor_token.size(0);
    if (factor < 1) {
        throw std::runtime_error("upsample_2d_fourier expects factor >= 1");
    }
    return factor;
}

void ValidateFactorTokenForMeta(const at::Tensor &factor_token) {
    if (factor_token.dim() != 1) {
        throw std::runtime_error("upsample_2d_fourier expects 1D factor_token");
    }
    if (factor_token.scalar_type() != at::kFloat) {
        throw std::runtime_error(
            "upsample_2d_fourier expects float32 factor_token");
    }
}

at::Tensor Upsample2DFourierCpu(const at::Tensor &x,
                                const at::Tensor &factor_token) {
    if (!x.device().is_cpu()) {
        throw std::runtime_error("upsample_2d_fourier expects CPU input");
    }
    if (x.dim() != 2) {
        throw std::runtime_error("upsample_2d_fourier expects 2D input");
    }
    if (x.scalar_type() != at::kFloat && x.scalar_type() != at::kDouble) {
        throw std::runtime_error(
            "upsample_2d_fourier expects float32/float64 input");
    }

    const int64_t factor = ValidateFactorToken(factor_token);

    const int64_t h = x.size(0);
    const int64_t w = x.size(1);
    const int64_t out_h = h * factor;
    const int64_t out_w = w * factor;

    const at::Tensor spec = at::fft_fft2(x);
    const at::Tensor spec_centered = at::fft_fftshift(spec);

    const int64_t pad_h = out_h - h;
    const int64_t pad_w = out_w - w;
    const int64_t pad_top = pad_h / 2;
    const int64_t pad_bottom = pad_h - pad_top;
    const int64_t pad_left = pad_w / 2;
    const int64_t pad_right = pad_w - pad_left;

    const std::vector<int64_t> pads = {pad_left, pad_right, pad_top,
                                       pad_bottom};
    const at::Tensor spec_padded = at::pad(spec_centered, pads);
    const at::Tensor spec_out = at::fft_ifftshift(spec_padded);

    const at::Tensor out = at::fft_ifft2(spec_out);
    const at::Tensor real = at::real(out);
    return real * (factor * factor);
}

at::Tensor Upsample2DFourierMeta(const at::Tensor &x,
                                 const at::Tensor &factor_token) {
    if (x.dim() != 2) {
        throw std::runtime_error("upsample_2d_fourier expects 2D input");
    }
    if (x.scalar_type() != at::kFloat && x.scalar_type() != at::kDouble) {
        throw std::runtime_error(
            "upsample_2d_fourier expects float32/float64 input");
    }

    ValidateFactorTokenForMeta(factor_token);

    const c10::SymInt factor = factor_token.sym_size(0);
    const c10::SymInt out_h = x.sym_size(0) * factor;
    const c10::SymInt out_w = x.sym_size(1) * factor;

    return at::empty_symint({out_h, out_w},
                            x.options().device(at::kMeta).requires_grad(false));
}

} // namespace kernel

TORCH_LIBRARY(ndarray, m) {
    m.def("upsample_2d_fourier(Tensor x, Tensor factor_token) -> Tensor");
}

TORCH_LIBRARY_IMPL(ndarray, CPU, m) {
    m.impl(kernel::kUpsampleOpName, kernel::Upsample2DFourierCpu);
}

TORCH_LIBRARY_IMPL(ndarray, Meta, m) {
    m.impl(kernel::kUpsampleOpName, kernel::Upsample2DFourierMeta);
}
