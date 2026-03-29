#include <ATen/dlpack.h>
#include <armadillo>
#include <cassert>
#include <cstring>
#include <iostream>

int main() {
    constexpr int64_t rows = 3;
    constexpr int64_t cols = 7;
    constexpr int64_t size = rows * cols;

    float data[size];
    for (int64_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    DLManagedTensor managed_tensor{};

    DLTensor &tensor = managed_tensor.dl_tensor;
    tensor.data = data;
    tensor.device = {kDLCPU, 0};
    tensor.ndim = 2;
    tensor.dtype = {kDLFloat, 32, 1};
    tensor.shape = new int64_t[2]{rows, cols};
    tensor.strides = new int64_t[2]{1, rows};
    tensor.byte_offset = 0;

    assert(tensor.strides[0] == 1);
    assert(tensor.strides[1] == rows);

    float *ptr = static_cast<float *>(tensor.data);
    arma::Mat<float> mat(ptr, rows, cols, false, false);

    assert(mat.n_rows == rows);
    assert(mat.n_cols == cols);
    assert(mat(1, 2) == 1 + 2 * rows);

    mat(0, 0) = 999.0f;
    assert(data[0] == 999.0f);

    delete[] tensor.shape;
    delete[] tensor.strides;

    std::cout << "All tests passed!\n";
    return 0;
}
