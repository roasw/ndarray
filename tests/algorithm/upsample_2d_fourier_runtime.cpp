#include <iostream>
#include <stdexcept>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/allclose.h>

#include "algorithm/upsample_2d_fourier.hpp"
#include "container/ndarray.hpp"

int main() {
    const char *package_path = nullptr;
#ifdef UPSAMPLE_2D_FOURIER_MODEL_PATH
    package_path = UPSAMPLE_2D_FOURIER_MODEL_PATH;
#endif
    if (package_path == nullptr) {
        throw std::runtime_error(
            "UPSAMPLE_2D_FOURIER_MODEL_PATH is not defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourier upsample(package_path, factor);
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1}, {1, 4}, {4, 1}, {3, 5}, {4, 6}, {7, 9}, {11, 13},
    };

    for (const auto [h, w] : shapes) {
        at::Tensor input_tensor =
            at::randn({h, w}, at::TensorOptions().dtype(at::kFloat));
        DLManagedTensor *input_dl = at::toDLPack(input_tensor);
        ndarray::ndarray<float> input =
            ndarray::ndarray<float>::FromDLPack(input_dl);

        ndarray::ndarray<float> output = upsample.Run(input);
        at::Tensor actual = at::fromDLPack(output.ToDLPack());

        const int64_t oh = h * factor;
        const int64_t ow = w * factor;
        if (actual.sizes() != at::IntArrayRef({oh, ow})) {
            throw std::runtime_error("AOTI runtime output has wrong shape");
        }

        using at::indexing::Slice;
        at::Tensor sampled = actual.index(
            {Slice(0, c10::nullopt, factor), Slice(0, c10::nullopt, factor)});
        if (!at::allclose(sampled, input_tensor, 1e-4, 1e-5)) {
            throw std::runtime_error(
                "AOTI runtime failed frequency-domain upsample identity check");
        }
    }

    std::cout << "AOTI runtime C++ test passed on variable 2D inputs\n";
    return 0;
}
