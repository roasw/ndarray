#include <cstdint>
#include <stdexcept>

#include <ATen/ATen.h>
#include <ATen/ops/fft_fft2.h>
#include <ATen/ops/fft_ifft2.h>
#include <ATen/ops/real.h>
#include <ATen/ops/zeros.h>
#include <torch/library.h>

namespace kernel {

namespace {

constexpr const char *kUpsampleOpName = "upsample_2d_fourier";

enum class FactorTokenValidationMode {
    Strict,
    Relaxed,
};

} // namespace

void ValidateFactorToken(const at::Tensor &factorToken,
                         FactorTokenValidationMode validationMode) {
    if (factorToken.dim() != 1) {
        throw std::runtime_error("upsample_2d_fourier expects 1D factor_token");
    }

    if (validationMode == FactorTokenValidationMode::Strict) {
        const int64_t factor = factorToken.size(0);
        if (factor < 1) {
            throw std::runtime_error("upsample_2d_fourier expects factor >= 1");
        }
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

    ValidateFactorToken(factorToken, FactorTokenValidationMode::Strict);
    // factor_token is a 1D tensor whose length == upsampling factor.
    // Only size(0) is read -- element values are never used.
    // This indirection exists because torch.export requires all
    // dynamic parameters as tensor inputs.
    const int64_t factor = factorToken.size(0);

    const int64_t h = x.size(0);
    const int64_t w = x.size(1);
    const int64_t outH = h * factor;
    const int64_t outW = w * factor;

    const at::Tensor spec = at::fft_fft2(x);

    const at::Tensor specPadded = at::zeros({outH, outW}, spec.options());

    const int64_t hPos = (h + 1) / 2;
    const int64_t wPos = (w + 1) / 2;

    specPadded.slice(0, 0, hPos).slice(1, 0, wPos) =
        spec.slice(0, 0, hPos).slice(1, 0, wPos);
    specPadded.slice(0, outH - (h - hPos), outH).slice(1, 0, wPos) =
        spec.slice(0, hPos, h).slice(1, 0, wPos);
    specPadded.slice(0, 0, hPos).slice(1, outW - (w - wPos), outW) =
        spec.slice(0, 0, hPos).slice(1, wPos, w);
    specPadded.slice(0, outH - (h - hPos), outH)
        .slice(1, outW - (w - wPos), outW) =
        spec.slice(0, hPos, h).slice(1, wPos, w);

    const at::Tensor out = at::fft_ifft2(specPadded);
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

    ValidateFactorToken(factorToken, FactorTokenValidationMode::Relaxed);

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
