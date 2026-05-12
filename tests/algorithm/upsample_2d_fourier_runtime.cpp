#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "algorithm/upsample_2d_fourier.hpp"
#include "upsample_2d_fourier_runtime_test_common.hpp"

int main() {
    std::string metadata_path;
#ifdef UPSAMPLE_2D_FOURIER_METADATA_PATH
    metadata_path = UPSAMPLE_2D_FOURIER_METADATA_PATH;
#endif
    if (metadata_path.empty()) {
        throw std::runtime_error(
            "UPSAMPLE_2D_FOURIER_METADATA_PATH is not defined");
    }

    const int64_t factor = 4;
    algorithm::Upsample2DFourier upsample(metadata_path, factor);
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1}, {1, 4}, {4, 1}, {3, 5}, {4, 6}, {7, 9}, {11, 13},
    };

    upsample_2d_fourier_test::RunTypedCases<float>(upsample, shapes, factor,
                                                   "AOTI runtime");
    upsample_2d_fourier_test::RunTypedCases<double>(upsample, shapes, factor,
                                                    "AOTI runtime");

    std::cout << "AOTI runtime C++ test passed on variable 2D inputs\n";
    return 0;
}
