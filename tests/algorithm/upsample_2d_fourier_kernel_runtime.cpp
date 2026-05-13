#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "algorithm/upsample_2d_fourier_kernel.hpp"
#include "upsample_2d_fourier_runtime_test_common.hpp"

int main() {
    std::string metadataPath;
#ifdef UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH
    metadataPath = UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH;
#endif
    if (metadataPath.empty()) {
        throw std::runtime_error(
            "UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH is not defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourierKernel upsample(metadataPath, factor);

    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1}, {1, 4}, {4, 1}, {3, 5}, {4, 6}, {7, 9}, {11, 13},
    };

    Upsample2DFourierTest::RunTypedCases<float>(upsample, shapes, factor,
                                                "AOTI kernel runtime");
    Upsample2DFourierTest::RunTypedCases<double>(upsample, shapes, factor,
                                                 "AOTI kernel runtime");

    std::cout << "AOTI kernel runtime C++ test passed on variable 2D inputs\n";
    return 0;
}
