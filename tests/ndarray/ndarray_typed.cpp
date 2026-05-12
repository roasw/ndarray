#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/dlpack.h>
#include <armadillo>

#include "container/ndarray.hpp"

namespace {

template <typename T> struct TypeName;
template <> struct TypeName<float> {
    static constexpr std::string_view value = "float";
};
template <> struct TypeName<double> {
    static constexpr std::string_view value = "double";
};

template <typename T> constexpr T Epsilon();
template <> constexpr float Epsilon<float>() { return 1e-5f; }
template <> constexpr double Epsilon<double>() { return 1e-10; }

template <typename T> constexpr uint8_t DtypeBits();
template <> constexpr uint8_t DtypeBits<float>() { return 32; }
template <> constexpr uint8_t DtypeBits<double>() { return 64; }

template <typename T>
void require_near(T a, T b, const std::string &msg, T eps = Epsilon<T>()) {
    if (std::fabs(a - b) > eps) {
        throw std::runtime_error(
            msg + " (" + std::to_string(static_cast<double>(a)) +
            " != " + std::to_string(static_cast<double>(b)) + ")");
    }
}

inline void require(bool cond, const std::string &msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

struct Counters {
    int passed = 0;
    int failed = 0;
};

template <typename Fn>
void RunCase(const std::string &name, Counters &counters, Fn fn) {
    try {
        fn();
        std::cout << "  PASS  " << name << "\n";
        ++counters.passed;
    } catch (const std::exception &e) {
        std::cout << "  FAIL  " << name << " - " << e.what() << "\n";
        ++counters.failed;
    } catch (...) {
        std::cout << "  FAIL  " << name << " - unknown exception\n";
        ++counters.failed;
    }
}

template <typename T> void TestDefaultConstruction() {
    ndarray::ndarray<T> a;
    require(a.GetNdim() == 0, "default ndim != 0");
    require(a.GetData() == nullptr, "default data != nullptr");
    require(a.GetShape().empty(), "default shape not empty");
    require(a.GetStrides().empty(), "default strides not empty");
}

template <typename T> void Test2dConstructionAndStrides() {
    ndarray::ndarray<T> a({3, 5});
    require(a.GetNdim() == 2, "2D ndim");
    require(a.GetShape() == (std::vector<int64_t>{3, 5}), "2D shape");
    require(a.GetStrides() == (std::vector<int64_t>{1, 3}),
            "2D col-major strides");
    require(a.GetData() != nullptr, "2D data null");
}

template <typename T> void TestCopySharesBuffer() {
    ndarray::ndarray<T> a({2, 2});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(1);

    ndarray::ndarray<T> b = a;
    require(b.GetData() == a.GetData(), "copy should share data pointer");

    b.At(int64_t(0), int64_t(0)) = static_cast<T>(99);
    require_near(a.At(int64_t(0), int64_t(0)), static_cast<T>(99),
                 "mutation through copy should be visible");
}

template <typename T> void TestCloneIndependentBuffer() {
    ndarray::ndarray<T> a({3, 3});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(5);

    ndarray::ndarray<T> b = a.Clone();
    require(b.GetData() != a.GetData(),
            "clone should have distinct data pointer");
    require_near(b.At(int64_t(0), int64_t(0)), static_cast<T>(5),
                 "clone should copy values");

    b.At(int64_t(0), int64_t(0)) = static_cast<T>(99);
    require_near(a.At(int64_t(0), int64_t(0)), static_cast<T>(5),
                 "mutation of clone should not affect original");
}

template <typename T> void TestAt2dColumnMajorLayout() {
    ndarray::ndarray<T> a({3, 4});
    T *data = a.GetData();
    for (int64_t i = 0; i < 12; ++i) {
        data[i] = static_cast<T>(i);
    }

    require_near(a.At(int64_t(0), int64_t(0)), static_cast<T>(0),
                 "col-major (0,0)");
    require_near(a.At(int64_t(1), int64_t(0)), static_cast<T>(1),
                 "col-major (1,0)");
    require_near(a.At(int64_t(2), int64_t(0)), static_cast<T>(2),
                 "col-major (2,0)");
    require_near(a.At(int64_t(0), int64_t(1)), static_cast<T>(3),
                 "col-major (0,1)");
}

template <typename T> void TestTransposeValues() {
    ndarray::ndarray<T> a({2, 3});
    for (int64_t r = 0; r < 2; ++r) {
        for (int64_t c = 0; c < 3; ++c) {
            a.At(r, c) = static_cast<T>(r * 3 + c);
        }
    }

    ndarray::ndarray<T> t = a.Transpose();
    for (int64_t r = 0; r < 2; ++r) {
        for (int64_t c = 0; c < 3; ++c) {
            require_near(t.At(c, r), a.At(r, c), "transpose value");
        }
    }
}

template <typename T> void TestArmaViewSharesData() {
    ndarray::ndarray<T> a({3, 4});
    a.At(int64_t(1), int64_t(2)) = static_cast<T>(42);

    auto view = a.ToArmadilloView();
    require_near(view.mat(1, 2), static_cast<T>(42),
                 "arma view reads ndarray data");
}

template <typename T> void TestDlpackZeroCopyAndMetadata() {
    ndarray::ndarray<T> a({3, 4});
    DLManagedTensor *mt = a.ToDLPack();
    require(mt != nullptr, "ToDLPack returned null");

    require(mt->dl_tensor.data == a.GetData(),
            "DLPack data pointer must equal ndarray data pointer");
    require(mt->dl_tensor.ndim == 2, "DLPack ndim");
    require(mt->dl_tensor.shape[0] == 3, "DLPack shape[0]");
    require(mt->dl_tensor.shape[1] == 4, "DLPack shape[1]");
    require(mt->dl_tensor.strides[0] == 1, "DLPack strides[0]");
    require(mt->dl_tensor.strides[1] == 3, "DLPack strides[1]");
    require(mt->dl_tensor.dtype.code == kDLFloat, "DLPack dtype code");
    require(mt->dl_tensor.dtype.bits == DtypeBits<T>(), "DLPack dtype bits");
    require(mt->dl_tensor.device.device_type == kDLCPU, "DLPack device");

    mt->deleter(mt);
}

template <typename T> void TestArithmeticAddMultiply() {
    ndarray::ndarray<T> a({2, 2});
    ndarray::ndarray<T> b({2, 2});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(1);
    a.At(int64_t(1), int64_t(1)) = static_cast<T>(3);
    b.At(int64_t(0), int64_t(0)) = static_cast<T>(10);
    b.At(int64_t(1), int64_t(1)) = static_cast<T>(5);

    ndarray::ndarray<T> c = a + b;
    require_near(c.At(int64_t(0), int64_t(0)), static_cast<T>(11), "add (0,0)");

    ndarray::ndarray<T> d = a * b;
    require_near(d.At(int64_t(1), int64_t(1)), static_cast<T>(15),
                 "multiply (1,1)");
}

template <typename T> void TestFromDlpackRoundtrip() {
    ndarray::ndarray<T> a({2, 3});
    a.At(int64_t(1), int64_t(2)) = static_cast<T>(7);
    DLManagedTensor *mt = a.ToDLPack();
    ndarray::ndarray<T> b = ndarray::ndarray<T>::FromDLPack(mt);

    require_near(b.At(int64_t(1), int64_t(2)), static_cast<T>(7),
                 "FromDLPack roundtrip value");
    b.At(int64_t(0), int64_t(0)) = static_cast<T>(123);
    require_near(a.At(int64_t(0), int64_t(0)), static_cast<T>(123),
                 "FromDLPack should preserve zero-copy behavior");
}

template <typename T> void TestTorchFromDlpackZeroCopy() {
    ndarray::ndarray<T> a({2, 3});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(3);

    at::Tensor t = at::fromDLPack(a.ToDLPack());
    require(t.data_ptr() == a.GetData(),
            "torch::fromDLPack data pointer must alias ndarray data");

    T *ptr = t.data_ptr<T>();
    ptr[0] = static_cast<T>(17);
    require_near(a.At(int64_t(0), int64_t(0)), static_cast<T>(17),
                 "torch write-through should update ndarray");

    a.At(int64_t(1), int64_t(0)) = static_cast<T>(29);
    require_near(ptr[1], static_cast<T>(29),
                 "ndarray write-through should update torch view");
}

template <typename T> void RunTypedSuite(Counters &counters) {
    const std::string prefix =
        std::string("[") + std::string(TypeName<T>::value) + "] ";
    RunCase(prefix + "default_construction", counters,
            [] { TestDefaultConstruction<T>(); });
    RunCase(prefix + "construction_and_strides", counters,
            [] { Test2dConstructionAndStrides<T>(); });
    RunCase(prefix + "copy_shares_buffer", counters,
            [] { TestCopySharesBuffer<T>(); });
    RunCase(prefix + "clone_independent_buffer", counters,
            [] { TestCloneIndependentBuffer<T>(); });
    RunCase(prefix + "at_column_major_layout", counters,
            [] { TestAt2dColumnMajorLayout<T>(); });
    RunCase(prefix + "transpose_values", counters,
            [] { TestTransposeValues<T>(); });
    RunCase(prefix + "arma_view_shares_data", counters,
            [] { TestArmaViewSharesData<T>(); });
    RunCase(prefix + "dlpack_zero_copy_and_metadata", counters,
            [] { TestDlpackZeroCopyAndMetadata<T>(); });
    RunCase(prefix + "arithmetic_add_multiply", counters,
            [] { TestArithmeticAddMultiply<T>(); });
    RunCase(prefix + "from_dlpack_roundtrip", counters,
            [] { TestFromDlpackRoundtrip<T>(); });
    RunCase(prefix + "torch_from_dlpack_zero_copy", counters,
            [] { TestTorchFromDlpackZeroCopy<T>(); });
}

} // namespace

int main() {
    std::cout << "=== ndarray typed tests ===\n";

    Counters counters;
    RunTypedSuite<float>(counters);
    RunTypedSuite<double>(counters);

    std::cout << "\n"
              << counters.passed << " passed, " << counters.failed
              << " failed.\n";
    return counters.failed > 0 ? 1 : 0;
}
