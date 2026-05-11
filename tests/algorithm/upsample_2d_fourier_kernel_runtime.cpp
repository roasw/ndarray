#include <iostream>
#include <stdexcept>
#include <vector>

#include "algorithm/upsample_2d_fourier_kernel.hpp"
#include "upsample_2d_fourier_runtime_test_common.hpp"

int main() {
    const char *package_path_f32 = nullptr;
    const char *package_path_f64 = nullptr;
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH
    package_path_f32 = UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH;
#endif
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH
    package_path_f64 = UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH;
#endif
    if (package_path_f32 == nullptr || package_path_f64 == nullptr) {
        throw std::runtime_error(
            "UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH or "
            "UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH is not defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourierKernel upsample(package_path_f32,
                                                package_path_f64, factor);

    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1}, {1, 4}, {4, 1}, {3, 5}, {4, 6}, {7, 9}, {11, 13},
    };

    upsample_2d_fourier_test::RunTypedCases<float>(upsample, shapes, factor,
                                                   "AOTI kernel runtime");
    upsample_2d_fourier_test::RunTypedCases<double>(upsample, shapes, factor,
                                                    "AOTI kernel runtime");

    std::cout << "AOTI kernel runtime C++ test passed on variable 2D inputs\n";
    return 0;
}
