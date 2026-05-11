#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/ops/ones.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "algorithm/detail/aoti_metadata_resolver.hpp"
#include "algorithm/upsample_2d_fourier.hpp"

namespace algorithm {

Upsample2DFourier::Upsample2DFourier(std::string metadata_path,
                                     int64_t upsample_factor)
    : m_upsampleFactor(upsample_factor) {
    if (m_upsampleFactor < 1) {
        throw std::runtime_error("Upsample factor must be >= 1");
    }

    m_paths = detail::ResolveTypedPackagePaths(metadata_path,
                                               detail::FileStem(__FILE__));
}

bool Upsample2DFourier::SupportsInputShape(const std::vector<int64_t> &shape) {
    if (shape.size() != 2) {
        return false;
    }
    const int64_t h = shape[0];
    const int64_t w = shape[1];
    return h > 0 && w > 0;
}

namespace {

template <typename T> struct UpsampleTraits;

template <> struct UpsampleTraits<float> {
    static constexpr at::ScalarType kScalarType = at::kFloat;
    static constexpr const char *kDTypeName = "float32";
};

template <> struct UpsampleTraits<double> {
    static constexpr at::ScalarType kScalarType = at::kDouble;
    static constexpr const char *kDTypeName = "float64";
};

} // namespace

template <typename T>
ndarray::ndarray<T>
Upsample2DFourier::RunTyped(const ndarray::ndarray<T> &input) const {
    const std::string &package_path = m_paths.SelectPath<T>(input.GetDevice());

    DLManagedTensor *input_dl = input.ToDLPack();
    if (!input_dl) {
        throw std::runtime_error("Upsample input cannot be empty");
    }

    if (!SupportsInputShape(input.GetShape())) {
        throw std::runtime_error(
            "Upsample input must be 2D with positive shape");
    }

    at::Tensor input_tensor = at::fromDLPack(input_dl);
    if (input_tensor.scalar_type() != UpsampleTraits<T>::kScalarType) {
        throw std::runtime_error(std::string("Upsample input must be ") +
                                 UpsampleTraits<T>::kDTypeName);
    }
    if (input_tensor.dim() != 2) {
        throw std::runtime_error("Upsample input must be a 2D tensor");
    }

    at::Tensor factor_token =
        at::ones({m_upsampleFactor}, at::TensorOptions().dtype(at::kFloat));

    auto package = torch::inductor::AOTIModelPackageLoader(package_path);
    std::vector<at::Tensor> outputs = package.run({input_tensor, factor_token});
    if (outputs.size() != 1) {
        throw std::runtime_error(
            "Upsample model must return exactly one output");
    }

    at::Tensor output_tensor = outputs[0];
    if (output_tensor.scalar_type() != UpsampleTraits<T>::kScalarType) {
        output_tensor = output_tensor.to(UpsampleTraits<T>::kScalarType);
    }
    DLManagedTensor *output_dl = at::toDLPack(output_tensor);
    return ndarray::ndarray<T>::FromDLPack(output_dl);
}

ndarray::ndarray<float>
Upsample2DFourier::Run(const ndarray::ndarray<float> &input) const {
    return RunTyped<float>(input);
}

ndarray::ndarray<double>
Upsample2DFourier::Run(const ndarray::ndarray<double> &input) const {
    return RunTyped<double>(input);
}

} // namespace algorithm
