#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/ops/ones.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "algorithm/detail/aoti_metadata_resolver.hpp"
#include "algorithm/upsample_2d_fourier_kernel.hpp"

namespace algorithm {

Upsample2DFourierKernel::Upsample2DFourierKernel(
    std::string package_path_float, std::string package_path_double,
    int64_t upsample_factor)
    : m_packagePathFloat(std::move(package_path_float)),
      m_packagePathDouble(std::move(package_path_double)),
      m_upsampleFactor(upsample_factor) {
    if (m_upsampleFactor < 1) {
        throw std::runtime_error("Upsample factor must be >= 1");
    }
}

Upsample2DFourierKernel::Upsample2DFourierKernel(std::string metadata_path,
                                                 int64_t upsample_factor)
    : m_upsampleFactor(upsample_factor) {
    if (m_upsampleFactor < 1) {
        throw std::runtime_error("Upsample factor must be >= 1");
    }

    detail::TypedPackagePaths paths = detail::ResolveTypedPackagePaths(
        metadata_path, "upsample_2d_fourier_kernel_cpu_f32_model",
        "upsample_2d_fourier_kernel_cpu_f64_model");
    m_packagePathFloat = std::move(paths.float_path);
    m_packagePathDouble = std::move(paths.double_path);
}

bool Upsample2DFourierKernel::SupportsInputShape(
    const std::vector<int64_t> &shape) {
    if (shape.size() != 2) {
        return false;
    }
    return shape[0] > 0 && shape[1] > 0;
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
Upsample2DFourierKernel::RunTyped(const ndarray::ndarray<T> &input) const {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Upsample2DFourierKernel supports float and double only");

    const std::string &package_path = [&]() -> const std::string & {
        if constexpr (std::is_same_v<T, float>) {
            return m_packagePathFloat;
        }
        return m_packagePathDouble;
    }();

    if (package_path.empty()) {
        if constexpr (std::is_same_v<T, float>) {
            throw std::runtime_error("Upsample float package path is empty");
        }
        throw std::runtime_error("Upsample double package path is empty");
    }

    if (!SupportsInputShape(input.GetShape())) {
        throw std::runtime_error(
            "Upsample input must be 2D with positive shape");
    }

    DLManagedTensor *input_dl = input.ToDLPack();
    if (!input_dl) {
        throw std::runtime_error("Upsample input cannot be empty");
    }

    at::Tensor input_tensor = at::fromDLPack(input_dl);
    if (input_tensor.scalar_type() != UpsampleTraits<T>::kScalarType) {
        throw std::runtime_error(std::string("Upsample input must be ") +
                                 UpsampleTraits<T>::kDTypeName);
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
Upsample2DFourierKernel::Run(const ndarray::ndarray<float> &input) const {
    return RunTyped<float>(input);
}

ndarray::ndarray<double>
Upsample2DFourierKernel::Run(const ndarray::ndarray<double> &input) const {
    return RunTyped<double>(input);
}

} // namespace algorithm
