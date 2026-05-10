#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "algorithm/upsample.hpp"

namespace algorithm {

Upsample::Upsample(std::string package_path)
    : m_packagePath(std::move(package_path)) {}

bool Upsample::SupportsInputShape(const std::vector<int64_t> &shape) {
    if (shape.size() != 2) {
        return false;
    }
    const int64_t h = shape[0];
    const int64_t w = shape[1];
    return h > 0 && w > 0;
}

ndarray::ndarray<float>
Upsample::Run(const ndarray::ndarray<float> &input) const {
    if (m_packagePath.empty()) {
        throw std::runtime_error("Upsample package path is empty");
    }

    DLManagedTensor *input_dl = input.ToDLPack();
    if (!input_dl) {
        throw std::runtime_error("Upsample input cannot be empty");
    }

    if (!SupportsInputShape(input.GetShape())) {
        throw std::runtime_error(
            "Upsample input must be 2D with positive shape");
    }

    at::Tensor input_tensor = at::fromDLPack(input_dl);
    if (input_tensor.scalar_type() != at::kFloat) {
        throw std::runtime_error("Upsample input must be float32");
    }
    if (input_tensor.dim() != 2) {
        throw std::runtime_error("Upsample input must be a 2D tensor");
    }

    auto package = torch::inductor::AOTIModelPackageLoader(m_packagePath);
    std::vector<at::Tensor> outputs = package.run({input_tensor});
    if (outputs.size() != 1) {
        throw std::runtime_error(
            "Upsample model must return exactly one output");
    }

    at::Tensor output_tensor = outputs[0];
    if (output_tensor.scalar_type() != at::kFloat) {
        output_tensor = output_tensor.to(at::kFloat);
    }
    DLManagedTensor *output_dl = at::toDLPack(output_tensor);
    return ndarray::ndarray<float>::FromDLPack(output_dl);
}

} // namespace algorithm
