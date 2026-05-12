#pragma once

#include <string>
#include <vector>

#include <ATen/dlpack.h>

#include "algorithm/detail/aoti_metadata_resolver.hpp"
#include "container/ndarray.hpp"

namespace algorithm {

class Upsample2DFourierKernel {
  public:
    explicit Upsample2DFourierKernel(const std::string &metadata_path,
                                     int64_t upsample_factor = 2);

    ndarray::ndarray<float> Run(const ndarray::ndarray<float> &input) const;
    ndarray::ndarray<double> Run(const ndarray::ndarray<double> &input) const;

    static bool SupportsInputShape(const std::vector<int64_t> &shape);

  private:
    template <typename T>
    ndarray::ndarray<T> RunTyped(const ndarray::ndarray<T> &input) const;

    detail::TypedPackagePaths m_paths;
    int64_t m_upsampleFactor;
};

} // namespace algorithm
