#pragma once

#include <string>
#include <vector>

#include <ATen/dlpack.h>

#include "container/ndarray.hpp"

namespace algorithm {

class Upsample2DFourierKernel {
  public:
    explicit Upsample2DFourierKernel(std::string metadata_path,
                                     int64_t upsample_factor = 2);

    explicit Upsample2DFourierKernel(std::string package_path_float,
                                     std::string package_path_double,
                                     int64_t upsample_factor = 2);

    ndarray::ndarray<float> Run(const ndarray::ndarray<float> &input) const;
    ndarray::ndarray<double> Run(const ndarray::ndarray<double> &input) const;

    static bool SupportsInputShape(const std::vector<int64_t> &shape);

  private:
    template <typename T>
    ndarray::ndarray<T> RunTyped(const ndarray::ndarray<T> &input) const;

    std::string m_packagePathFloat;
    std::string m_packagePathDouble;
    int64_t m_upsampleFactor;
};

} // namespace algorithm
