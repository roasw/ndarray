#pragma once

#include <string>

#include <ATen/dlpack.h>

#include "container/ndarray.hpp"

namespace algorithm {

class UpsamplePt2 {
  public:
    explicit UpsamplePt2(std::string package_path);

    ndarray::ndarray<float> Run(const ndarray::ndarray<float> &input) const;

  private:
    std::string m_packagePath;
};

} // namespace algorithm
