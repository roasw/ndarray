#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "algorithm/upsample_pt2.hpp"

namespace algorithm {

UpsamplePt2::UpsamplePt2(std::string package_path)
    : m_packagePath(std::move(package_path)) {}

ndarray::ndarray<float>
UpsamplePt2::Run(const ndarray::ndarray<float> &input) const {
    if (m_packagePath.empty()) {
        throw std::runtime_error("Upsample package path is empty");
    }

    DLManagedTensor *input_dl = input.ToDLPack();
    if (!input_dl) {
        throw std::runtime_error("Upsample input cannot be empty");
    }

    at::Tensor input_tensor = at::fromDLPack(input_dl);

    auto package = torch::inductor::AOTIModelPackageLoader(m_packagePath);
    std::vector<at::Tensor> outputs = package.run({input_tensor});
    if (outputs.size() != 1) {
        throw std::runtime_error(
            "Upsample model must return exactly one output");
    }

    at::Tensor output_tensor = outputs[0];
    DLManagedTensor *output_dl = at::toDLPack(output_tensor);
    return ndarray::ndarray<float>::FromDLPack(output_dl);
}

} // namespace algorithm
