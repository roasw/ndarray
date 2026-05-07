#include <cstddef>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/upsample_bilinear2d.h>

#include "algorithm/upsample_pt2.hpp"
#include "container/ndarray.hpp"

int main() {
    const char *package_path = nullptr;
#ifdef UPSAMPLE_PT2_PATH
    package_path = UPSAMPLE_PT2_PATH;
#endif
    if (package_path == nullptr) {
        throw std::runtime_error("UPSAMPLE_PT2_PATH is not defined");
    }

    at::Tensor input_tensor =
        at::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({1, 1, 2, 2});
    DLManagedTensor *input_dl = at::toDLPack(input_tensor);
    ndarray::ndarray<float> input =
        ndarray::ndarray<float>::FromDLPack(input_dl);

    algorithm::UpsamplePt2 upsample(package_path);
    ndarray::ndarray<float> output = upsample.Run(input);

    at::Tensor expected = at::upsample_bilinear2d(input_tensor, {4, 4}, false,
                                                  std::nullopt, std::nullopt);
    at::Tensor actual = at::fromDLPack(output.ToDLPack());

    if (!at::allclose(actual, expected, 1e-5, 1e-6)) {
        throw std::runtime_error("AOTI runtime output mismatch");
    }

    std::cout << "AOTI runtime C++ test passed without Python runtime\n";
    return 0;
}
