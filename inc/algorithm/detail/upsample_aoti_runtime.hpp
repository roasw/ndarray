#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/ops/ones.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "algorithm/detail/aoti_metadata_resolver.hpp"
#include "container/ndarray.hpp"

namespace algorithm::detail {

template <typename T> struct UpsampleTraits;

template <> struct UpsampleTraits<float> {
    static constexpr at::ScalarType kScalarType = at::kFloat;
    static constexpr std::string_view kDTypeName = "float32";
};

template <> struct UpsampleTraits<double> {
    static constexpr at::ScalarType kScalarType = at::kDouble;
    static constexpr std::string_view kDTypeName = "float64";
};

inline bool SupportsUpsampleInputShape(const std::vector<int64_t> &shape) {
    if (shape.size() != 2) {
        return false;
    }
    return shape[0] > 0 && shape[1] > 0;
}

template <typename T>
ndarray::ndarray<T> RunUpsampleAoti(const ndarray::ndarray<T> &input,
                                    const TypedPackagePaths &paths,
                                    int64_t upsampleFactor) {
    const std::string &packagePath = paths.SelectPath<T>(input.GetDevice());

    if (!SupportsUpsampleInputShape(input.GetShape())) {
        throw std::runtime_error(
            "Upsample input must be 2D with positive shape");
    }

    DLManagedTensor *inputDl = input.ToDLPack();
    if (!inputDl) {
        throw std::runtime_error("Upsample input cannot be empty");
    }

    at::Tensor inputTensor = at::fromDLPack(inputDl);
    if (inputTensor.scalar_type() != UpsampleTraits<T>::kScalarType) {
        throw std::runtime_error(std::string("Upsample input must be ") +
                                 std::string(UpsampleTraits<T>::kDTypeName));
    }
    if (inputTensor.dim() != 2) {
        throw std::runtime_error("Upsample input must be a 2D tensor");
    }

    // factor_token is a 1D tensor whose length == upsampling factor.
    // The element values are never read; only size(0) is used by the
    // exported graph (via sym_size.int).  This indirection exists because
    // torch.export requires all dynamic parameters as tensor inputs.
    // int64 matches the Python export trace -- keeping them in sync is
    // required for AOTIModelPackageLoader::run() to accept the token.
    at::Tensor factorToken =
        at::ones({upsampleFactor}, at::TensorOptions().dtype(at::kLong));

    auto package = torch::inductor::AOTIModelPackageLoader(packagePath);
    std::vector<at::Tensor> outputs = package.run({inputTensor, factorToken});
    if (outputs.size() != 1) {
        throw std::runtime_error(
            "Upsample model must return exactly one output");
    }

    at::Tensor outputTensor = outputs[0];
    if (outputTensor.scalar_type() != UpsampleTraits<T>::kScalarType) {
        outputTensor = outputTensor.to(UpsampleTraits<T>::kScalarType);
    }

    return ndarray::ndarray<T>::FromDLPack(at::toDLPack(outputTensor));
}

} // namespace algorithm::detail
