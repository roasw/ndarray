#include <stdexcept>
#include <string>
#include <vector>

#include "algorithm/detail/aoti_metadata_resolver.hpp"
#include "algorithm/detail/upsample_aoti_runtime.hpp"
#include "algorithm/upsample_2d_fourier_kernel.hpp"

namespace algorithm {

Upsample2DFourierKernel::Upsample2DFourierKernel(
    const std::string &metadata_path, int64_t upsample_factor)
    : m_upsampleFactor(upsample_factor) {
    if (m_upsampleFactor < 1) {
        throw std::runtime_error("Upsample factor must be >= 1");
    }

    m_paths = detail::ResolveTypedPackagePaths(metadata_path,
                                               detail::FileStem(__FILE__));
}

bool Upsample2DFourierKernel::SupportsInputShape(
    const std::vector<int64_t> &shape) {
    return detail::SupportsUpsampleInputShape(shape);
}

template <typename T>
ndarray::ndarray<T>
Upsample2DFourierKernel::RunTyped(const ndarray::ndarray<T> &input) const {
    return detail::RunUpsampleAoti<T>(input, m_paths, m_upsampleFactor);
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
