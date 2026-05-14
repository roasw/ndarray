#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/Tensor.h>
#include <ATen/dlpack.h>
#include <armadillo>

#include "container/ndarray.hpp"
#include "test_common.hpp"

namespace {

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

template <typename T>
void RequireNear(T a, T b, const std::string &msg,
                 double eps = NdarrayTestCommon::Epsilon<T>()) {
    if (NdarrayTestCommon::DiffMagnitude(a, b) > eps) {
        throw std::runtime_error(msg + " (" + ScalarToString(a) +
                                 " != " + ScalarToString(b) + ")");
    }
}

inline void Require(bool cond, const std::string &msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

/**
 * @brief Verify that default construction creates an empty ndarray sentinel.
 *
 * @details This test checks the contract for an uninitialized value-like
 * container: rank is zero, data pointer is null, and shape/stride metadata is
 * empty. The check works because these accessors are direct reflections of the
 * internal `m_tensor == nullptr` state, so any non-empty metadata would imply
 * accidental allocation.
 */
template <typename T> void TestDefaultConstruction() {
    ndarray::ndarray<T> a;
    Require(a.GetNdim() == 0, "default ndim != 0");
    Require(a.GetData() == nullptr, "default data != nullptr");
    Require(a.GetShape().empty(), "default shape not empty");
    Require(a.GetStrides().empty(), "default strides not empty");
}

/**
 * @brief Validate shape and column-major stride initialization for 2D arrays.
 *
 * @details The test constructs a `{3,5}` ndarray and asserts strides `{1,3}`.
 * This works because the implementation computes strides cumulatively in
 * column-major order, so the first axis has unit stride and the second axis is
 * multiplied by the row count.
 */
template <typename T> void Test2dConstructionAndStrides() {
    ndarray::ndarray<T> a({3, 5});
    Require(a.GetNdim() == 2, "2D ndim");
    Require(a.GetShape() == (std::vector<int64_t>{3, 5}), "2D shape");
    Require(a.GetStrides() == (std::vector<int64_t>{1, 3}),
            "2D col-major strides");
    Require(a.GetData() != nullptr, "2D data null");
}

/**
 * @brief Ensure copy construction preserves zero-copy shared ownership.
 *
 * @details This test checks pointer equality and write-through behavior between
 * source and copied ndarray. It proves correctness because both objects should
 * hold the same shared `DLManagedTensor` owner; mutating one must immediately
 * affect the other if no hidden copy happened.
 */
template <typename T> void TestCopySharesBuffer() {
    ndarray::ndarray<T> a({2, 2});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(1);

    ndarray::ndarray<T> b = a;
    Require(b.GetData() == a.GetData(), "copy should share data pointer");

    b.At(int64_t(0), int64_t(0)) = static_cast<T>(99);
    RequireNear(a.At(int64_t(0), int64_t(0)), static_cast<T>(99),
                "mutation through copy should be visible");
}

/**
 * @brief Ensure Clone performs a deep copy with independent storage.
 *
 * @details The test asserts pointer inequality and verifies value equality
 * first, then mutates the clone and confirms the original is unchanged. The
 * sequence proves both copy correctness and storage independence.
 */
template <typename T> void TestCloneIndependentBuffer() {
    ndarray::ndarray<T> a({3, 3});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(5);

    ndarray::ndarray<T> b = a.Clone();
    Require(b.GetData() != a.GetData(),
            "clone should have distinct data pointer");
    RequireNear(b.At(int64_t(0), int64_t(0)), static_cast<T>(5),
                "clone should copy values");

    b.At(int64_t(0), int64_t(0)) = static_cast<T>(99);
    RequireNear(a.At(int64_t(0), int64_t(0)), static_cast<T>(5),
                "mutation of clone should not affect original");
}

/**
 * @brief Verify `At(r,c)` indexing against raw column-major memory layout.
 *
 * @details Data are written linearly through `GetData()` and then sampled by
 * two-dimensional indices. Matching expected values validates that index-to-
 * offset conversion uses stored DLPack strides in column-major order.
 */
template <typename T> void TestAt2dColumnMajorLayout() {
    ndarray::ndarray<T> a({3, 4});
    T *data = a.GetData();
    for (int64_t i = 0; i < 12; ++i) {
        data[i] = static_cast<T>(i);
    }

    RequireNear(a.At(int64_t(0), int64_t(0)), static_cast<T>(0),
                "col-major (0,0)");
    RequireNear(a.At(int64_t(1), int64_t(0)), static_cast<T>(1),
                "col-major (1,0)");
    RequireNear(a.At(int64_t(2), int64_t(0)), static_cast<T>(2),
                "col-major (2,0)");
    RequireNear(a.At(int64_t(0), int64_t(1)), static_cast<T>(3),
                "col-major (0,1)");
}

/**
 * @brief Validate 2D transpose value mapping.
 *
 * @details The test fills a rectangular matrix with unique values and checks
 * `t(c,r) == a(r,c)` for all coordinates. This proves transpose correctness
 * independent of backend (Armadillo path for non-bool and explicit loop for
 * bool).
 */
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
            RequireNear(t.At(c, r), a.At(r, c), "transpose value");
        }
    }
}

/**
 * @brief Confirm Armadillo view is a zero-copy alias for non-bool types.
 *
 * @details The ndarray is written first, then read via `arma::Mat` view.
 * Successful read-through demonstrates that the view references the same
 * storage rather than a copied buffer.
 */
template <typename T> void TestArmaViewSharesData() {
    ndarray::ndarray<T> a({3, 4});
    a.At(int64_t(1), int64_t(2)) = static_cast<T>(42);

    auto view = a.ToArmadilloView();
    RequireNear(view.mat(1, 2), static_cast<T>(42),
                "arma view reads ndarray data");
}

/**
 * @brief Ensure bool specialization rejects Armadillo view requests.
 *
 * @details Armadillo bool matrices are intentionally unsupported in this
 * project. The test expects `ToArmadilloView()` to throw, confirming the
 * explicit guard remains in place.
 */
template <> void TestArmaViewSharesData<bool>() {
    ndarray::ndarray<bool> a({3, 4});
    bool threw = false;
    try {
        (void)a.ToArmadilloView();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    Require(threw, "bool ToArmadilloView should throw");
}

/**
 * @brief Validate DLPack export metadata and zero-copy pointer aliasing.
 *
 * @details The test checks pointer aliasing, ndim/shape/stride consistency,
 * dtype code/bits, and CPU device metadata on `ToDLPack()`. This works because
 * the exporter should only wrap existing storage and clone metadata arrays,
 * never duplicate the tensor payload.
 */
template <typename T> void TestDlpackZeroCopyAndMetadata() {
    ndarray::ndarray<T> a({3, 4});
    DLManagedTensor *mt = a.ToDLPack();
    Require(mt != nullptr, "ToDLPack returned null");

    Require(mt->dl_tensor.data == a.GetData(),
            "DLPack data pointer must equal ndarray data pointer");
    Require(mt->dl_tensor.ndim == 2, "DLPack ndim");
    Require(mt->dl_tensor.shape[0] == 3, "DLPack shape[0]");
    Require(mt->dl_tensor.shape[1] == 4, "DLPack shape[1]");
    Require(mt->dl_tensor.strides[0] == 1, "DLPack strides[0]");
    Require(mt->dl_tensor.strides[1] == 3, "DLPack strides[1]");
    Require(mt->dl_tensor.dtype.code == DtypeCode<T>(),
            "DLPack dtype code expected " +
                std::to_string(static_cast<int>(DtypeCode<T>())) + ", got " +
                std::to_string(static_cast<int>(mt->dl_tensor.dtype.code)));
    Require(mt->dl_tensor.dtype.bits == DtypeBits<T>(), "DLPack dtype bits");
    Require(mt->dl_tensor.device.device_type == kDLCPU, "DLPack device");

    mt->deleter(mt);
}

/**
 * @brief Verify `FromDLPack(ToDLPack())` roundtrip keeps storage shared.
 *
 * @details A value written in the source ndarray is observed in the roundtrip
 * ndarray, then write-through from roundtrip back to source is verified. This
 * proves adoption of the same underlying storage with proper ownership.
 */
template <typename T> void TestFromDlpackRoundtrip() {
    ndarray::ndarray<T> a({2, 3});
    a.At(int64_t(1), int64_t(2)) = static_cast<T>(7);
    DLManagedTensor *mt = a.ToDLPack();
    ndarray::ndarray<T> b = ndarray::ndarray<T>::FromDLPack(mt);

    RequireNear(b.At(int64_t(1), int64_t(2)), static_cast<T>(7),
                "FromDLPack roundtrip value");
    b.At(int64_t(0), int64_t(0)) = static_cast<T>(123);
    RequireNear(a.At(int64_t(0), int64_t(0)), static_cast<T>(123),
                "FromDLPack should preserve zero-copy behavior");
}

/**
 * @brief Confirm bidirectional zero-copy aliasing between ndarray and torch.
 *
 * @details The test creates a torch tensor from ndarray DLPack, checks pointer
 * equality, mutates from torch and ndarray sides, and validates both views see
 * updates. This demonstrates stable shared storage across the bridge.
 */
template <typename T> void TestTorchFromDlpackZeroCopy() {
    ndarray::ndarray<T> a({2, 3});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(3);

    at::Tensor t = at::fromDLPack(a.ToDLPack());
    Require(t.data_ptr() == a.GetData(),
            "torch::fromDLPack data pointer must alias ndarray data");

    T *ptr = reinterpret_cast<T *>(t.data_ptr());
    ptr[0] = static_cast<T>(17);
    RequireNear(a.At(int64_t(0), int64_t(0)), static_cast<T>(17),
                "torch write-through should update ndarray");

    a.At(int64_t(1), int64_t(0)) = static_cast<T>(29);
    RequireNear(ptr[1], static_cast<T>(29),
                "ndarray write-through should update torch view");
}

/**
 * @brief Stress repeated DLPack roundtrips for aliasing stability.
 *
 * @details The loop repeatedly exports and re-adopts storage 128 times while
 * checking pointer equality and write-through behavior each iteration. This
 * catches ownership regressions that may only appear after repeated hand-offs.
 */
template <typename T> void TestRepeatedFromDlpackRoundtripZeroCopy() {
    ndarray::ndarray<T> a({2, 3});
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(1);

    for (int64_t i = 0; i < 128; ++i) {
        ndarray::ndarray<T> b = ndarray::ndarray<T>::FromDLPack(a.ToDLPack());
        Require(b.GetData() == a.GetData(),
                "repeated FromDLPack must preserve aliasing");

        b.At(int64_t(0), int64_t(0)) = static_cast<T>(i);
        RequireNear(
            a.At(int64_t(0), int64_t(0)), static_cast<T>(i),
            "repeated FromDLPack write-through must preserve zero-copy");
    }
}

template <typename T>
/**
 * @brief Verify adopted DLPack storage outlives the original ndarray object.
 *
 * @details A temporary source ndarray exports DLPack and is destroyed, while
 * the adopted ndarray remains in scope. Continued read/write correctness after
 * source destruction proves shared-owner lifetime management is correct.
 */
void TestFromDlpackKeepsStorageAliveAfterSourceDestroyed() {
    ndarray::ndarray<T> survivor;
    T *sourceData = nullptr;

    {
        ndarray::ndarray<T> source({2, 3});
        source.At(int64_t(0), int64_t(1)) = static_cast<T>(11);
        sourceData = source.GetData();
        survivor = ndarray::ndarray<T>::FromDLPack(source.ToDLPack());
    }

    Require(survivor.GetData() == sourceData,
            "FromDLPack survivor must keep source storage alive");
    RequireNear(survivor.At(int64_t(0), int64_t(1)), static_cast<T>(11),
                "value must remain readable after source destruction");

    survivor.At(int64_t(1), int64_t(2)) = static_cast<T>(37);
    RequireNear(survivor.At(int64_t(1), int64_t(2)), static_cast<T>(37),
                "survivor must remain writable after source destruction");
}

/**
 * @brief Verify torch view lifetime when source ndarray is temporary.
 *
 * @details A torch tensor is created from a temporary ndarray DLPack capsule,
 * then used after the ndarray goes out of scope. Subsequent roundtrip back to
 * ndarray confirms the storage remains alive and mutable through torch.
 */
template <typename T> void TestTorchViewOutlivesTemporaryNdarray() {
    at::Tensor t;
    {
        ndarray::ndarray<T> tmp({2, 3});
        tmp.At(int64_t(0), int64_t(0)) = static_cast<T>(5);
        t = at::fromDLPack(tmp.ToDLPack());
        Require(t.data_ptr() == tmp.GetData(),
                "torch tensor must alias temporary ndarray storage");
    }

    T *ptr = reinterpret_cast<T *>(t.data_ptr());
    RequireNear(ptr[0], static_cast<T>(5),
                "torch tensor must remain valid after ndarray destruction");

    ptr[1] = static_cast<T>(19);
    ndarray::ndarray<T> b = ndarray::ndarray<T>::FromDLPack(at::toDLPack(t));
    RequireNear(b.At(int64_t(1), int64_t(0)), static_cast<T>(19),
                "torch mutation must remain visible through ndarray roundtrip");
}

template <typename T>
void RunTypedSuite(NdarrayTestCommon::Counters &counters) {
    const std::string prefix =
        std::string("[") + std::string(NdarrayTestCommon::TypeName<T>::value) +
        "] ";
    NdarrayTestCommon::RunCase(prefix + "default_construction", counters,
                               [] { TestDefaultConstruction<T>(); });
    NdarrayTestCommon::RunCase(prefix + "construction_and_strides", counters,
                               [] { Test2dConstructionAndStrides<T>(); });
    NdarrayTestCommon::RunCase(prefix + "copy_shares_buffer", counters,
                               [] { TestCopySharesBuffer<T>(); });
    NdarrayTestCommon::RunCase(prefix + "clone_independent_buffer", counters,
                               [] { TestCloneIndependentBuffer<T>(); });
    NdarrayTestCommon::RunCase(prefix + "at_column_major_layout", counters,
                               [] { TestAt2dColumnMajorLayout<T>(); });
    NdarrayTestCommon::RunCase(prefix + "transpose_values", counters,
                               [] { TestTransposeValues<T>(); });
    NdarrayTestCommon::RunCase(prefix + "arma_view_shares_data", counters,
                               [] { TestArmaViewSharesData<T>(); });
    NdarrayTestCommon::RunCase(prefix + "dlpack_zero_copy_and_metadata",
                               counters,
                               [] { TestDlpackZeroCopyAndMetadata<T>(); });
    NdarrayTestCommon::RunCase(prefix + "from_dlpack_roundtrip", counters,
                               [] { TestFromDlpackRoundtrip<T>(); });
    NdarrayTestCommon::RunCase(prefix + "torch_from_dlpack_zero_copy", counters,
                               [] { TestTorchFromDlpackZeroCopy<T>(); });
    NdarrayTestCommon::RunCase(
        prefix + "repeated_from_dlpack_roundtrip_zero_copy", counters,
        [] { TestRepeatedFromDlpackRoundtripZeroCopy<T>(); });
    NdarrayTestCommon::RunCase(
        prefix + "from_dlpack_keeps_storage_alive", counters,
        [] { TestFromDlpackKeepsStorageAliveAfterSourceDestroyed<T>(); });
    NdarrayTestCommon::RunCase(
        prefix + "torch_view_outlives_temporary_ndarray", counters,
        [] { TestTorchViewOutlivesTemporaryNdarray<T>(); });
}

} // namespace

int main() {
    std::cout << "=== ndarray typed tests ===\n";

    NdarrayTestCommon::Counters counters;
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
