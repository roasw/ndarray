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
        at::Tensor inputTensor = at::randn(
            {h, w}, at::TensorOptions().dtype(
                        Upsample2DFourierTest::TestConfig<T>::kDType));

        DLManagedTensor *inputDl = at::toDLPack(inputTensor);
        ndarray::ndarray<T> input = ndarray::ndarray<T>::FromDLPack(inputDl);

        ndarray::ndarray<T> referenceOutput = reference.Run(input);
        ndarray::ndarray<T> kernelOutput = kernel.Run(input);

        at::Tensor actualReference = at::fromDLPack(referenceOutput.ToDLPack());
        at::Tensor actualKernel = at::fromDLPack(kernelOutput.ToDLPack());

        const int64_t outH = h * factor;
        const int64_t outW = w * factor;
        if (actualReference.sizes() != at::IntArrayRef({outH, outW})) {
            throw std::runtime_error(
                "Parity runtime reference output has wrong shape");
        }
        if (actualKernel.sizes() != at::IntArrayRef({outH, outW})) {
            throw std::runtime_error(
                "Parity runtime kernel output has wrong shape");
        }

        if (!actualReference.isfinite().all().item<bool>()) {
            throw std::runtime_error(
                "Parity runtime reference output has NaN/Inf");
        }
        if (!actualKernel.isfinite().all().item<bool>()) {
            throw std::runtime_error(
                "Parity runtime kernel output has NaN/Inf");
        }

        if (!at::allclose(actualKernel, actualReference,
                          Upsample2DFourierTest::TestConfig<T>::kRtol,
                          Upsample2DFourierTest::TestConfig<T>::kAtol)) {
            const std::string typeName(
                Upsample2DFourierTest::TestConfig<T>::kTypeName);
            throw std::runtime_error(
                std::string("Parity runtime outputs mismatch for ") + typeName);
        }
    }
}

} // namespace

int main() {
    std::string referenceMetadataPath;
    std::string kernelMetadataPath;

#ifdef UPSAMPLE_2D_FOURIER_METADATA_PATH
    referenceMetadataPath = UPSAMPLE_2D_FOURIER_METADATA_PATH;
#endif
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH
    kernelMetadataPath = UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH;
#endif

    if (referenceMetadataPath.empty() || kernelMetadataPath.empty()) {
        throw std::runtime_error(
            "Parity runtime metadata macros are not fully defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourier reference(referenceMetadataPath, factor);
    algorithm::Upsample2DFourierKernel kernel(kernelMetadataPath, factor);

    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1}, {1, 4}, {4, 1}, {3, 5}, {4, 6}, {7, 9}, {11, 13},
    };

    CompareTypedOutputs<float>(reference, kernel, shapes, factor);
    CompareTypedOutputs<double>(reference, kernel, shapes, factor);

    std::cout
        << "Parity runtime C++ test passed for standard vs kernel paths\n";
    return 0;
}
