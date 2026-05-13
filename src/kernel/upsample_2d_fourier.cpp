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

int64_t ValidateFactorToken(const at::Tensor &factorToken) {
    if (factorToken.dim() != 1) {
        throw std::runtime_error("upsample_2d_fourier expects 1D factor_token");
    }

    const int64_t factor = factorToken.size(0);
    if (factor < 1) {
        throw std::runtime_error("upsample_2d_fourier expects factor >= 1");
    }
    return factor;
}

void ValidateFactorTokenForMeta(const at::Tensor &factorToken) {
    if (factorToken.dim() != 1) {
        throw std::runtime_error("upsample_2d_fourier expects 1D factor_token");
    }
}

at::Tensor Upsample2DFourierCpu(const at::Tensor &x,
                                const at::Tensor &factorToken) {
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

    const int64_t factor = ValidateFactorToken(factorToken);

    const int64_t h = x.size(0);
    const int64_t w = x.size(1);
    const int64_t outH = h * factor;
    const int64_t outW = w * factor;

    const at::Tensor spec = at::fft_fft2(x);
    const at::Tensor specCentered = at::fft_fftshift(spec);

    const int64_t padH = outH - h;
    const int64_t padW = outW - w;
    const int64_t padTop = padH / 2;
    const int64_t padBottom = padH - padTop;
    const int64_t padLeft = padW / 2;
    const int64_t padRight = padW - padLeft;

    const std::vector<int64_t> pads = {padLeft, padRight, padTop, padBottom};
    const at::Tensor specPadded = at::pad(specCentered, pads);
    const at::Tensor specOut = at::fft_ifftshift(specPadded);

    const at::Tensor out = at::fft_ifft2(specOut);
    const at::Tensor real = at::real(out);
    return real * (factor * factor);
}

at::Tensor Upsample2DFourierMeta(const at::Tensor &x,
                                 const at::Tensor &factorToken) {
    if (x.dim() != 2) {
        throw std::runtime_error("upsample_2d_fourier expects 2D input");
    }
    if (x.scalar_type() != at::kFloat && x.scalar_type() != at::kDouble) {
        throw std::runtime_error(
            "upsample_2d_fourier expects float32/float64 input");
    }

    ValidateFactorTokenForMeta(factorToken);

    const c10::SymInt factor = factorToken.sym_size(0);
    const c10::SymInt outH = x.sym_size(0) * factor;
    const c10::SymInt outW = x.sym_size(1) * factor;

    return at::empty_symint({outH, outW},
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
