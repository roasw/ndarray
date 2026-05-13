#include <cmath>
#include <complex>
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
template <> struct TypeName<int32_t> {
    static constexpr std::string_view value = "int32";
};
template <> struct TypeName<int64_t> {
    static constexpr std::string_view value = "int64";
};
template <> struct TypeName<std::complex<float>> {
    static constexpr std::string_view value = "complex32";
};
template <> struct TypeName<std::complex<double>> {
    static constexpr std::string_view value = "complex64";
};
template <> struct TypeName<bool> {
    static constexpr std::string_view value = "bool";
};

template <typename T> constexpr double Epsilon();
template <> constexpr double Epsilon<float>() { return 1e-5; }
template <> constexpr double Epsilon<double>() { return 1e-10; }
template <> constexpr double Epsilon<int32_t>() { return 0.0; }
template <> constexpr double Epsilon<int64_t>() { return 0.0; }
template <> constexpr double Epsilon<std::complex<float>>() { return 1e-5; }
template <> constexpr double Epsilon<std::complex<double>>() { return 1e-10; }
template <> constexpr double Epsilon<bool>() { return 0.0; }

template <typename T> constexpr uint8_t DtypeBits();
template <> constexpr uint8_t DtypeBits<float>() { return 32; }
template <> constexpr uint8_t DtypeBits<double>() { return 64; }
template <> constexpr uint8_t DtypeBits<int32_t>() { return 32; }
template <> constexpr uint8_t DtypeBits<int64_t>() { return 64; }
template <> constexpr uint8_t DtypeBits<std::complex<float>>() { return 64; }
template <> constexpr uint8_t DtypeBits<std::complex<double>>() { return 128; }
template <> constexpr uint8_t DtypeBits<bool>() { return 8; }

template <typename T> constexpr uint8_t DtypeCode();
template <> constexpr uint8_t DtypeCode<float>() { return kDLFloat; }
template <> constexpr uint8_t DtypeCode<double>() { return kDLFloat; }
template <> constexpr uint8_t DtypeCode<int32_t>() { return kDLInt; }
template <> constexpr uint8_t DtypeCode<int64_t>() { return kDLInt; }
template <> constexpr uint8_t DtypeCode<std::complex<float>>() {
    return kDLComplex;
}
template <> constexpr uint8_t DtypeCode<std::complex<double>>() {
    return kDLComplex;
}
template <> constexpr uint8_t DtypeCode<bool>() { return kDLBool; }

template <typename T> std::string ScalarToString(const T &value) {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return std::to_string(static_cast<double>(value.real())) + "+" +
               std::to_string(static_cast<double>(value.imag())) + "i";
    } else {
        return std::to_string(static_cast<double>(value));
    }
}

template <typename T> double DiffMagnitude(T a, T b) {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return std::abs(a - b);
    } else {
        return std::fabs(static_cast<double>(a) - static_cast<double>(b));
    }
}

template <typename T>
void require_near(T a, T b, const std::string &msg, double eps = Epsilon<T>()) {
    if (DiffMagnitude(a, b) > eps) {
        throw std::runtime_error(msg + " (" + ScalarToString(a) +
                                 " != " + ScalarToString(b) + ")");
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

template <> void TestArmaViewSharesData<bool>() {
    ndarray::ndarray<bool> a({3, 4});
    bool threw = false;
    try {
        (void)a.ToArmadilloView();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "bool ToArmadilloView should throw");
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
    require(mt->dl_tensor.dtype.code == DtypeCode<T>(),
            "DLPack dtype code expected " +
                std::to_string(static_cast<int>(DtypeCode<T>())) + ", got " +
                std::to_string(static_cast<int>(mt->dl_tensor.dtype.code)));
    require(mt->dl_tensor.dtype.bits == DtypeBits<T>(), "DLPack dtype bits");
    require(mt->dl_tensor.device.device_type == kDLCPU, "DLPack device");

    mt->deleter(mt);
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

    T *ptr = reinterpret_cast<T *>(t.data_ptr());
    ptr[0] = static_cast<T>(17);
    require_near(a.At(int64_t(0), int64_t(0)), static_cast<T>(17),
                 "torch write-through should update ndarray");

    a.At(int64_t(1), int64_t(0)) = static_cast<T>(29);
    require_near(ptr[1], static_cast<T>(29),
                 "ndarray write-through should update torch view");
}

template <typename T> void TestRepeatedFromDlpackRoundtripZeroCopy() {
    ndarray::ndarray<T> a({2, 3});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(1);

    for (int64_t i = 0; i < 128; ++i) {
        ndarray::ndarray<T> b = ndarray::ndarray<T>::FromDLPack(a.ToDLPack());
        require(b.GetData() == a.GetData(),
                "repeated FromDLPack must preserve aliasing");

        b.At(int64_t(0), int64_t(0)) = static_cast<T>(i);
        require_near(
            a.At(int64_t(0), int64_t(0)), static_cast<T>(i),
            "repeated FromDLPack write-through must preserve zero-copy");
    }
}

template <typename T>
void TestFromDlpackKeepsStorageAliveAfterSourceDestroyed() {
    ndarray::ndarray<T> survivor;
    T *sourceData = nullptr;

    {
        ndarray::ndarray<T> source({2, 3});
        source.At(int64_t(0), int64_t(1)) = static_cast<T>(11);
        sourceData = source.GetData();
        survivor = ndarray::ndarray<T>::FromDLPack(source.ToDLPack());
    }

    require(survivor.GetData() == sourceData,
            "FromDLPack survivor must keep source storage alive");
    require_near(survivor.At(int64_t(0), int64_t(1)), static_cast<T>(11),
                 "value must remain readable after source destruction");

    survivor.At(int64_t(1), int64_t(2)) = static_cast<T>(37);
    require_near(survivor.At(int64_t(1), int64_t(2)), static_cast<T>(37),
                 "survivor must remain writable after source destruction");
}

template <typename T> void TestTorchViewOutlivesTemporaryNdarray() {
    at::Tensor t;
    {
        ndarray::ndarray<T> tmp({2, 3});
        tmp.At(int64_t(0), int64_t(0)) = static_cast<T>(5);
        t = at::fromDLPack(tmp.ToDLPack());
        require(t.data_ptr() == tmp.GetData(),
                "torch tensor must alias temporary ndarray storage");
    }

    T *ptr = reinterpret_cast<T *>(t.data_ptr());
    require_near(ptr[0], static_cast<T>(5),
                 "torch tensor must remain valid after ndarray destruction");

    ptr[1] = static_cast<T>(19);
    ndarray::ndarray<T> b = ndarray::ndarray<T>::FromDLPack(at::toDLPack(t));
    require_near(
        b.At(int64_t(1), int64_t(0)), static_cast<T>(19),
        "torch mutation must remain visible through ndarray roundtrip");
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
    RunCase(prefix + "from_dlpack_roundtrip", counters,
            [] { TestFromDlpackRoundtrip<T>(); });
    RunCase(prefix + "torch_from_dlpack_zero_copy", counters,
            [] { TestTorchFromDlpackZeroCopy<T>(); });
    RunCase(prefix + "repeated_from_dlpack_roundtrip_zero_copy", counters,
            [] { TestRepeatedFromDlpackRoundtripZeroCopy<T>(); });
    RunCase(prefix + "from_dlpack_keeps_storage_alive", counters,
            [] { TestFromDlpackKeepsStorageAliveAfterSourceDestroyed<T>(); });
    RunCase(prefix + "torch_view_outlives_temporary_ndarray", counters,
            [] { TestTorchViewOutlivesTemporaryNdarray<T>(); });
}

} // namespace

int main() {
    std::cout << "=== ndarray typed tests ===\n";

    Counters counters;
    RunTypedSuite<float>(counters);
    RunTypedSuite<double>(counters);
    RunTypedSuite<int32_t>(counters);
    RunTypedSuite<int64_t>(counters);
    RunTypedSuite<std::complex<float>>(counters);
    RunTypedSuite<std::complex<double>>(counters);
    RunTypedSuite<bool>(counters);

    std::cout << "\n"
              << counters.passed << " passed, " << counters.failed
              << " failed.\n";
    return counters.failed > 0 ? 1 : 0;
}
