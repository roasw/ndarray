#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/ops/allclose.h>

#include "algorithm/upsample_2d_fourier.hpp"
#include "algorithm/upsample_2d_fourier_kernel.hpp"
#include "upsample_2d_fourier_runtime_test_common.hpp"

namespace {

template <typename T>
void CompareTypedOutputs(algorithm::Upsample2DFourier &reference,
                         algorithm::Upsample2DFourierKernel &kernel,
                         const std::vector<std::pair<int64_t, int64_t>> &shapes,
                         int64_t factor) {
    for (const auto [h, w] : shapes) {
        at::Tensor input_tensor = at::randn(
            {h, w}, at::TensorOptions().dtype(
                        upsample_2d_fourier_test::TestConfig<T>::kDType));

        DLManagedTensor *input_dl = at::toDLPack(input_tensor);
        ndarray::ndarray<T> input = ndarray::ndarray<T>::FromDLPack(input_dl);

        ndarray::ndarray<T> reference_output = reference.Run(input);
        ndarray::ndarray<T> kernel_output = kernel.Run(input);

        at::Tensor actual_reference =
            at::fromDLPack(reference_output.ToDLPack());
        at::Tensor actual_kernel = at::fromDLPack(kernel_output.ToDLPack());

        const int64_t out_h = h * factor;
        const int64_t out_w = w * factor;
        if (actual_reference.sizes() != at::IntArrayRef({out_h, out_w})) {
            throw std::runtime_error(
                "Parity runtime reference output has wrong shape");
        }
        if (actual_kernel.sizes() != at::IntArrayRef({out_h, out_w})) {
            throw std::runtime_error(
                "Parity runtime kernel output has wrong shape");
        }

        if (!actual_reference.isfinite().all().item<bool>()) {
            throw std::runtime_error(
                "Parity runtime reference output has NaN/Inf");
        }
        if (!actual_kernel.isfinite().all().item<bool>()) {
            throw std::runtime_error(
                "Parity runtime kernel output has NaN/Inf");
        }

        if (!at::allclose(actual_kernel, actual_reference,
                          upsample_2d_fourier_test::TestConfig<T>::kRtol,
                          upsample_2d_fourier_test::TestConfig<T>::kAtol)) {
            const std::string type_name(
                upsample_2d_fourier_test::TestConfig<T>::kTypeName);
            throw std::runtime_error(
                std::string("Parity runtime outputs mismatch for ") +
                type_name);
        }
    }
}

} // namespace

int main() {
    std::string reference_metadata_path;
    std::string kernel_metadata_path;

#ifdef UPSAMPLE_2D_FOURIER_METADATA_PATH
    reference_metadata_path = UPSAMPLE_2D_FOURIER_METADATA_PATH;
#endif
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH
    kernel_metadata_path = UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH;
#endif

    if (reference_metadata_path.empty() || kernel_metadata_path.empty()) {
        throw std::runtime_error(
            "Parity runtime metadata macros are not fully defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourier reference(reference_metadata_path, factor);
    algorithm::Upsample2DFourierKernel kernel(kernel_metadata_path, factor);

    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1}, {1, 4}, {4, 1}, {3, 5}, {4, 6}, {7, 9}, {11, 13},
    };

    CompareTypedOutputs<float>(reference, kernel, shapes, factor);
    CompareTypedOutputs<double>(reference, kernel, shapes, factor);

    std::cout
        << "Parity runtime C++ test passed for standard vs kernel paths\n";
    return 0;
}
