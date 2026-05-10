#pragma once

#include <string>
#include <vector>

#include <ATen/dlpack.h>

#include "container/ndarray.hpp"

namespace algorithm {

class Upsample {
  public:
    explicit Upsample(std::string package_path);

    ndarray::ndarray<float> Run(const ndarray::ndarray<float> &input) const;

    static bool SupportsInputShape(const std::vector<int64_t> &shape);

  private:
    std::string m_packagePath;
};

} // namespace algorithm
