#include <iostream>
#include <stdexcept>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/allclose.h>

#include "algorithm/upsample_2d_fourier_kernel.hpp"
#include "container/ndarray.hpp"

int main() {
    const char *package_path_f32 = nullptr;
    const char *package_path_f64 = nullptr;
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH
    package_path_f32 = UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH;
#endif
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH
    package_path_f64 = UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH;
#endif
    if (package_path_f32 == nullptr || package_path_f64 == nullptr) {
        throw std::runtime_error(
            "UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH or "
            "UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH is not defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourierKernel upsample(package_path_f32,
                                                package_path_f64, factor);

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
            throw std::runtime_error(
                "AOTI kernel runtime output has wrong shape");
        }
        if (!actual.isfinite().all().item<bool>()) {
            throw std::runtime_error(
                "AOTI kernel runtime float32 output has NaN/Inf");
        }

        if ((h % 2 == 0) && (w % 2 == 0)) {
            using at::indexing::Slice;
            at::Tensor sampled = actual.index({Slice(0, c10::nullopt, factor),
                                               Slice(0, c10::nullopt, factor)});
            if (!at::allclose(sampled, input_tensor, 1e-4, 1e-5)) {
                throw std::runtime_error(
                    "AOTI kernel runtime float32 identity check failed");
            }
        }

        at::Tensor input_tensor_f64 =
            at::randn({h, w}, at::TensorOptions().dtype(at::kDouble));
        DLManagedTensor *input_dl_f64 = at::toDLPack(input_tensor_f64);
        ndarray::ndarray<double> input_f64 =
            ndarray::ndarray<double>::FromDLPack(input_dl_f64);

        ndarray::ndarray<double> output_f64 = upsample.Run(input_f64);
        at::Tensor actual_f64 = at::fromDLPack(output_f64.ToDLPack());

        if (actual_f64.sizes() != at::IntArrayRef({oh, ow})) {
            throw std::runtime_error(
                "AOTI kernel runtime float64 output has wrong shape");
        }
        if (!actual_f64.isfinite().all().item<bool>()) {
            throw std::runtime_error(
                "AOTI kernel runtime float64 output has NaN/Inf");
        }

        if ((h % 2 == 0) && (w % 2 == 0)) {
            using at::indexing::Slice;
            at::Tensor sampled_f64 =
                actual_f64.index({Slice(0, c10::nullopt, factor),
                                  Slice(0, c10::nullopt, factor)});
            if (!at::allclose(sampled_f64, input_tensor_f64, 1e-10, 1e-10)) {
                throw std::runtime_error(
                    "AOTI kernel runtime float64 identity check failed");
            }
        }
    }

    std::cout << "AOTI kernel runtime C++ test passed on variable 2D inputs\n";
    return 0;
}
