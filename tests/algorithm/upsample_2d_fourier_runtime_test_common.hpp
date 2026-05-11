#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/allclose.h>

#include "container/ndarray.hpp"

namespace upsample_2d_fourier_test {

template <typename T> struct TestConfig;

template <> struct TestConfig<float> {
    static constexpr auto kDType = at::kFloat;
    static constexpr double kRtol = 1e-4;
    static constexpr double kAtol = 1e-5;
    static constexpr const char *kTypeName = "float32";
};

template <> struct TestConfig<double> {
    static constexpr auto kDType = at::kDouble;
    static constexpr double kRtol = 1e-10;
    static constexpr double kAtol = 1e-10;
    static constexpr const char *kTypeName = "float64";
};

template <typename T, typename TUpsample>
void RunTypedCases(TUpsample &upsample,
                   const std::vector<std::pair<int64_t, int64_t>> &shapes,
                   int64_t factor, const std::string &error_prefix) {
    for (const auto [h, w] : shapes) {
        at::Tensor input_tensor =
            at::randn({h, w}, at::TensorOptions().dtype(TestConfig<T>::kDType));
        DLManagedTensor *input_dl = at::toDLPack(input_tensor);
        ndarray::ndarray<T> input = ndarray::ndarray<T>::FromDLPack(input_dl);

        ndarray::ndarray<T> output = upsample.Run(input);
        at::Tensor actual = at::fromDLPack(output.ToDLPack());

        const int64_t oh = h * factor;
        const int64_t ow = w * factor;
        if (actual.sizes() != at::IntArrayRef({oh, ow})) {
            throw std::runtime_error(error_prefix + " " +
                                     TestConfig<T>::kTypeName +
                                     " output has wrong shape");
        }

        if (!actual.isfinite().all().item<bool>()) {
            throw std::runtime_error(error_prefix + " " +
                                     TestConfig<T>::kTypeName +
                                     " output has NaN/Inf");
        }

        if ((h % 2 == 0) && (w % 2 == 0)) {
            using at::indexing::Slice;
            at::Tensor sampled = actual.index({Slice(0, c10::nullopt, factor),
                                               Slice(0, c10::nullopt, factor)});
            if (!at::allclose(sampled, input_tensor, TestConfig<T>::kRtol,
                              TestConfig<T>::kAtol)) {
                throw std::runtime_error(error_prefix + " " +
                                         TestConfig<T>::kTypeName +
                                         " identity check failed");
            }
        }
    }
}

} // namespace upsample_2d_fourier_test
