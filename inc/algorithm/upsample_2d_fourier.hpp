#pragma once

#include <string>
#include <vector>

#include <ATen/dlpack.h>

#include "container/ndarray.hpp"

namespace algorithm {

class Upsample2DFourier {
  public:
    explicit Upsample2DFourier(std::string package_path,
                               int64_t upsample_factor = 2);

    ndarray::ndarray<float> Run(const ndarray::ndarray<float> &input) const;

    static bool SupportsInputShape(const std::vector<int64_t> &shape);

  private:
    std::string m_packagePath;
    int64_t m_upsampleFactor;
};

} // namespace algorithm
