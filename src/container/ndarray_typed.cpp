#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <ATen/ops/div.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_or.h>
#include <armadillo>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

#include "container/ndarray.hpp"

namespace ndarray {

namespace {

template <typename T> struct DTypeTraits;

template <> struct DTypeTraits<float> {
    static constexpr DLDataType kDType() { return {kDLFloat, 32, 1}; }
};

template <> struct DTypeTraits<double> {
    static constexpr DLDataType kDType() { return {kDLFloat, 64, 1}; }
};

template <> struct DTypeTraits<int32_t> {
    static constexpr DLDataType kDType() { return {kDLInt, 32, 1}; }
};

template <> struct DTypeTraits<int64_t> {
    static constexpr DLDataType kDType() { return {kDLInt, 64, 1}; }
};

template <> struct DTypeTraits<std::complex<float>> {
    static constexpr DLDataType kDType() { return {kDLComplex, 64, 1}; }
};

template <> struct DTypeTraits<std::complex<double>> {
    static constexpr DLDataType kDType() { return {kDLComplex, 128, 1}; }
};

template <> struct DTypeTraits<bool> {
    static constexpr DLDataType kDType() { return {kDLBool, 8, 1}; }
};

template <typename T> struct IsSupportedNdarrayType : std::false_type {};

template <> struct IsSupportedNdarrayType<float> : std::true_type {};
template <> struct IsSupportedNdarrayType<double> : std::true_type {};
template <> struct IsSupportedNdarrayType<int32_t> : std::true_type {};
template <> struct IsSupportedNdarrayType<int64_t> : std::true_type {};
template <>
struct IsSupportedNdarrayType<std::complex<float>> : std::true_type {};
template <>
struct IsSupportedNdarrayType<std::complex<double>> : std::true_type {};
template <> struct IsSupportedNdarrayType<bool> : std::true_type {};

int64_t NumElements(const DLTensor &t) {
    int64_t n = 1;
    for (int i = 0; i < t.ndim; ++i) {
        n *= t.shape[i];
    }
    return n;
}

DLDevice ToDLDevice(c10::DeviceType deviceType) {
    switch (deviceType) {
    case c10::DeviceType::CPU:
        return {kDLCPU, 0};
    case c10::DeviceType::CUDA:
        return {kDLCUDA, 0};
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

c10::DeviceType FromDLDevice(DLDeviceType deviceType) {
    switch (deviceType) {
    case kDLCPU:
        return c10::DeviceType::CPU;
    case kDLCUDA:
        return c10::DeviceType::CUDA;
    default:
        throw std::runtime_error("Unsupported DLDevice type");
    }
}

void OwningDeleter(DLManagedTensor *mt) {
    if (!mt) {
        return;
    }
    delete[] mt->dl_tensor.shape;
    delete[] mt->dl_tensor.strides;
    delete static_cast<c10::DataPtr *>(mt->manager_ctx);
    delete mt;
}

void ViewDeleter(DLManagedTensor *mt) {
    if (!mt) {
        return;
    }
    delete[] mt->dl_tensor.shape;
    delete[] mt->dl_tensor.strides;
    delete static_cast<std::shared_ptr<DLManagedTensor> *>(mt->manager_ctx);
    delete mt;
}

void ManagedTensorDeleter(DLManagedTensor *mt) {
    if (!mt) {
        return;
    }
    if (mt->deleter) {
        mt->deleter(mt);
        return;
    }

    delete[] mt->dl_tensor.shape;
    delete[] mt->dl_tensor.strides;
    delete mt;
}

template <typename T>
std::shared_ptr<DLManagedTensor>
AllocateTensor(const std::vector<int64_t> &shape, c10::DeviceType deviceType) {
    static_assert(IsSupportedNdarrayType<T>::value,
                  "ndarray only supports configured scalar dtypes");

    const int ndim = static_cast<int>(shape.size());
    const int64_t total = [&] {
        int64_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        return n;
    }();

    c10::Allocator *alloc = c10::GetAllocator(deviceType);
    c10::DataPtr dataPtr =
        alloc->allocate(total * static_cast<int64_t>(sizeof(T)));

    auto *mt = new DLManagedTensor();
    mt->dl_tensor.data = dataPtr.get();
    mt->dl_tensor.device = ToDLDevice(deviceType);
    mt->dl_tensor.ndim = ndim;
    mt->dl_tensor.dtype = DTypeTraits<T>::kDType();
    mt->dl_tensor.byte_offset = 0;

    int64_t *shapePtr = nullptr;
    int64_t *stridesPtr = nullptr;

    if (ndim > 0) {
        shapePtr = new int64_t[ndim];
        stridesPtr = new int64_t[ndim];

        for (int i = 0; i < ndim; ++i) {
            shapePtr[i] = shape[i];
        }

        stridesPtr[0] = 1;
        for (int i = 1; i < ndim; ++i) {
            stridesPtr[i] = stridesPtr[i - 1] * shape[i - 1];
        }
    }

    mt->dl_tensor.shape = shapePtr;
    mt->dl_tensor.strides = stridesPtr;
    mt->manager_ctx = new c10::DataPtr(std::move(dataPtr));
    mt->deleter = OwningDeleter;

    return {mt, OwningDeleter};
}

template <typename T> bool IsExpectedDType(const DLTensor &t) {
    const auto dtype = DTypeTraits<T>::kDType();
    return t.dtype.code == dtype.code && t.dtype.bits == dtype.bits &&
           t.dtype.lanes == dtype.lanes;
}

template <typename DataType, typename... Indices>
DataType &AtImpl(const DLTensor &tensor, DataType *data, Indices... indices) {
    static_assert((std::is_convertible_v<Indices, int64_t> && ...),
                  "Indices must be convertible to int64_t");

    constexpr size_t kCount = sizeof...(Indices);
    if constexpr (kCount == 0) {
        return *data;
    } else if constexpr (kCount == 1) {
        if (tensor.ndim != 1) {
            throw std::runtime_error("Invalid dimensions for At()");
        }
        const std::array<int64_t, 1> idx = {static_cast<int64_t>(indices)...};
        if (idx[0] < 0 || idx[0] >= tensor.shape[0]) {
            throw std::runtime_error("Index out of bounds for At()");
        }
        return data[idx[0] * tensor.strides[0]];
    } else if constexpr (kCount == 2) {
        if (tensor.ndim != 2) {
            throw std::runtime_error("Invalid dimensions for At()");
        }
        const std::array<int64_t, 2> idx = {static_cast<int64_t>(indices)...};
        if (idx[0] < 0 || idx[0] >= tensor.shape[0] || idx[1] < 0 ||
            idx[1] >= tensor.shape[1]) {
            throw std::runtime_error("Index out of bounds for At()");
        }
        return data[idx[0] * tensor.strides[0] + idx[1] * tensor.strides[1]];
    } else {
        static_assert(kCount <= 2, "At() supports up to 2 indices");
    }
}

template <typename T>
void ValidateBinaryOpInputs(const ndarray<T> &a, const ndarray<T> &b,
                            std::string_view opName) {
    if (!a.GetData() || !b.GetData()) {
        throw std::runtime_error(std::string("Cannot ") + std::string(opName) +
                                 " empty arrays");
    }
    const auto shapeA = a.GetShape();
    const auto shapeB = b.GetShape();
    if (shapeA.size() != 2 || shapeB.size() != 2) {
        throw std::runtime_error(std::string(opName) +
                                 " only supported for 2D arrays");
    }
    if (shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1]) {
        throw std::runtime_error(std::string("Shape mismatch for ") +
                                 std::string(opName));
    }
    if (a.GetDevice() != b.GetDevice()) {
        throw std::runtime_error(std::string("Device mismatch for ") +
                                 std::string(opName));
    }
}

template <typename T, typename Op>
ndarray<T> TorchBinaryOp(const ndarray<T> &a, const ndarray<T> &b,
                         std::string_view opName, Op op) {
    ValidateBinaryOpInputs(a, b, opName);

    at::Tensor lhs = at::fromDLPack(a.ToDLPack());
    at::Tensor rhs = at::fromDLPack(b.ToDLPack());
    at::Tensor computed = op(lhs, rhs);

    if (computed.scalar_type() != lhs.scalar_type()) {
        computed = computed.to(lhs.scalar_type());
    }

    ndarray<T> out(a.GetShape(), a.GetDevice());
    at::Tensor outTensor = at::fromDLPack(out.ToDLPack());
    outTensor.copy_(computed);

    return out;
}

} // namespace

template <typename T>
ndarray<T>::ndarray(std::shared_ptr<DLManagedTensor> tensor)
    : m_tensor(std::move(tensor)) {}

template <typename T> ndarray<T>::ndarray() : m_tensor(nullptr) {}

template <typename T>
ndarray<T>::ndarray(std::vector<int64_t> shape, c10::DeviceType deviceType)
    : m_tensor(AllocateTensor<T>(shape, deviceType)) {}

template <typename T> int64_t ndarray<T>::GetNdim() const {
    return m_tensor ? m_tensor->dl_tensor.ndim : 0;
}

template <typename T> std::vector<int64_t> ndarray<T>::GetShape() const {
    if (!m_tensor) {
        return {};
    }
    const auto &t = m_tensor->dl_tensor;
    return {t.shape, t.shape + t.ndim};
}

template <typename T> std::vector<int64_t> ndarray<T>::GetStrides() const {
    if (!m_tensor) {
        return {};
    }
    const auto &t = m_tensor->dl_tensor;
    return {t.strides, t.strides + t.ndim};
}

template <typename T> c10::DeviceType ndarray<T>::GetDevice() const {
    if (!m_tensor) {
        return c10::DeviceType::CPU;
    }
    return FromDLDevice(m_tensor->dl_tensor.device.device_type);
}

template <typename T> T *ndarray<T>::GetData() const {
    if (!m_tensor) {
        return nullptr;
    }
    return static_cast<T *>(m_tensor->dl_tensor.data);
}

template <typename T>
template <typename... Indices>
T &ndarray<T>::At(Indices... indices) {
    if (!m_tensor) {
        throw std::runtime_error("ndarray is empty");
    }

    const auto &t = m_tensor->dl_tensor;
    return AtImpl(t, static_cast<T *>(t.data), indices...);
}

template <typename T>
template <typename... Indices>
const T &ndarray<T>::At(Indices... indices) const {
    if (!m_tensor) {
        throw std::runtime_error("ndarray is empty");
    }

    const auto &t = m_tensor->dl_tensor;
    return AtImpl(t, static_cast<const T *>(t.data), indices...);
}

template <typename T> ndarray<T> ndarray<T>::Transpose() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Transpose only supported for 2D arrays");
    }

    const auto &t = m_tensor->dl_tensor;
    std::vector<int64_t> newShape = {t.shape[1], t.shape[0]};
    ndarray<T> result(newShape, GetDevice());

    if constexpr (std::is_same_v<T, bool>) {
        for (int64_t r = 0; r < t.shape[0]; ++r) {
            for (int64_t c = 0; c < t.shape[1]; ++c) {
                result.At(c, r) = At(r, c);
            }
        }
    } else {
        arma::Mat<T> src(GetData(), t.shape[0], t.shape[1], false, true);
        arma::Mat<T> dst(result.GetData(), result.m_tensor->dl_tensor.shape[0],
                         result.m_tensor->dl_tensor.shape[1], false, true);
        dst = src.t();
    }
    return result;
}

template <typename T> ndarray<T> ndarray<T>::Clone() const {
    if (!m_tensor) {
        return ndarray<T>();
    }

    const auto &t = m_tensor->dl_tensor;
    ndarray<T> result(GetShape(), GetDevice());
    std::copy_n(static_cast<const T *>(t.data), NumElements(t),
                result.GetData());
    return result;
}

template <typename T> ArmadilloView<T> ndarray<T>::ToArmadilloView() {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error(
            "ToArmadilloView only supported for 2D arrays");
    }

    if constexpr (std::is_same_v<T, bool>) {
        throw std::runtime_error("ToArmadilloView not supported for bool");
    } else {
        const auto &t = m_tensor->dl_tensor;
        return {arma::Mat<T>(static_cast<T *>(t.data), t.shape[0], t.shape[1],
                             false, true),
                m_tensor};
    }
}

template <typename T> ArmadilloView<T> ndarray<T>::ToArmadilloView() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error(
            "ToArmadilloView only supported for 2D arrays");
    }

    if constexpr (std::is_same_v<T, bool>) {
        throw std::runtime_error("ToArmadilloView not supported for bool");
    } else {
        const auto &t = m_tensor->dl_tensor;
        return {arma::Mat<T>(static_cast<T *>(t.data), t.shape[0], t.shape[1],
                             false, true),
                m_tensor};
    }
}

template <typename T> DLManagedTensor *ndarray<T>::ToDLPack() const {
    if (!m_tensor) {
        return nullptr;
    }

    const auto &src = m_tensor->dl_tensor;
    auto *mt = new DLManagedTensor();
    mt->dl_tensor.data = src.data;
    mt->dl_tensor.device = src.device;
    mt->dl_tensor.ndim = src.ndim;
    mt->dl_tensor.dtype = src.dtype;
    mt->dl_tensor.byte_offset = src.byte_offset;

    int64_t *shapePtr = nullptr;
    int64_t *stridesPtr = nullptr;

    if (src.ndim > 0) {
        shapePtr = new int64_t[src.ndim];
        stridesPtr = new int64_t[src.ndim];
        std::copy_n(src.shape, src.ndim, shapePtr);
        std::copy_n(src.strides, src.ndim, stridesPtr);
    }

    mt->dl_tensor.shape = shapePtr;
    mt->dl_tensor.strides = stridesPtr;
    mt->manager_ctx = new std::shared_ptr<DLManagedTensor>(m_tensor);
    mt->deleter = ViewDeleter;

    return mt;
}

template <typename T>
ndarray<T> ndarray<T>::FromDLPack(DLManagedTensor *managedTensor) {
    if (!managedTensor) {
        return ndarray<T>();
    }

    const auto &t = managedTensor->dl_tensor;
    if (!IsExpectedDType<T>(t)) {
        ManagedTensorDeleter(managedTensor);
        throw std::runtime_error("FromDLPack expects matching dtype");
    }

    return ndarray<T>(
        std::shared_ptr<DLManagedTensor>(managedTensor, ManagedTensorDeleter));
}

template <typename T>
ndarray<T> ndarray<T>::Add(const ndarray<T> &other) const {
    if constexpr (std::is_same_v<T, bool>) {
        return TorchBinaryOp(*this, other, "Add",
                             [](const at::Tensor &a, const at::Tensor &b) {
                                 return at::logical_or(a, b);
                             });
    } else {
        return TorchBinaryOp(
            *this, other, "Add",
            [](const at::Tensor &a, const at::Tensor &b) { return a + b; });
    }
}

template <typename T>
ndarray<T> ndarray<T>::Subtract(const ndarray<T> &other) const {
    if constexpr (std::is_same_v<T, bool>) {
        (void)other;
        throw std::runtime_error("Subtract not supported for bool arrays");
    } else {
        return TorchBinaryOp(
            *this, other, "Subtract",
            [](const at::Tensor &a, const at::Tensor &b) { return a - b; });
    }
}

template <typename T>
ndarray<T> ndarray<T>::Multiply(const ndarray<T> &other) const {
    if constexpr (std::is_same_v<T, bool>) {
        return TorchBinaryOp(*this, other, "Multiply",
                             [](const at::Tensor &a, const at::Tensor &b) {
                                 return at::logical_and(a, b);
                             });
    } else {
        return TorchBinaryOp(
            *this, other, "Multiply",
            [](const at::Tensor &a, const at::Tensor &b) { return a * b; });
    }
}

template <typename T>
ndarray<T> ndarray<T>::Divide(const ndarray<T> &other) const {
    if constexpr (std::is_same_v<T, bool>) {
        (void)other;
        throw std::runtime_error("Divide not supported for bool arrays");
    } else if constexpr (std::is_integral_v<T>) {
        return TorchBinaryOp(*this, other, "Divide",
                             [](const at::Tensor &a, const at::Tensor &b) {
                                 return at::div(a.to(at::kDouble),
                                                b.to(at::kDouble));
                             });
    } else {
        return TorchBinaryOp(*this, other, "Divide",
                             [](const at::Tensor &a, const at::Tensor &b) {
                                 return at::div(a, b);
                             });
    }
}

template <typename T>
ndarray<T> ndarray<T>::operator+(const ndarray<T> &other) const {
    return Add(other);
}

template <typename T>
ndarray<T> ndarray<T>::operator-(const ndarray<T> &other) const {
    return Subtract(other);
}

template <typename T>
ndarray<T> ndarray<T>::operator*(const ndarray<T> &other) const {
    return Multiply(other);
}

template <typename T>
ndarray<T> ndarray<T>::operator/(const ndarray<T> &other) const {
    return Divide(other);
}

template class ndarray<float>;
template class ndarray<double>;
template class ndarray<int32_t>;
template class ndarray<int64_t>;
template class ndarray<std::complex<float>>;
template class ndarray<std::complex<double>>;
template class ndarray<bool>;

template float &ndarray<float>::At();
template const float &ndarray<float>::At() const;
template float &ndarray<float>::At<int64_t>(int64_t);
template const float &ndarray<float>::At<int64_t>(int64_t) const;
template float &ndarray<float>::At<int64_t, int64_t>(int64_t, int64_t);
template const float &ndarray<float>::At<int64_t, int64_t>(int64_t,
                                                           int64_t) const;

template double &ndarray<double>::At();
template const double &ndarray<double>::At() const;
template double &ndarray<double>::At<int64_t>(int64_t);
template const double &ndarray<double>::At<int64_t>(int64_t) const;
template double &ndarray<double>::At<int64_t, int64_t>(int64_t, int64_t);
template const double &ndarray<double>::At<int64_t, int64_t>(int64_t,
                                                             int64_t) const;

template int32_t &ndarray<int32_t>::At();
template const int32_t &ndarray<int32_t>::At() const;
template int32_t &ndarray<int32_t>::At<int64_t>(int64_t);
template const int32_t &ndarray<int32_t>::At<int64_t>(int64_t) const;
template int32_t &ndarray<int32_t>::At<int64_t, int64_t>(int64_t, int64_t);
template const int32_t &ndarray<int32_t>::At<int64_t, int64_t>(int64_t,
                                                               int64_t) const;

template int64_t &ndarray<int64_t>::At();
template const int64_t &ndarray<int64_t>::At() const;
template int64_t &ndarray<int64_t>::At<int64_t>(int64_t);
template const int64_t &ndarray<int64_t>::At<int64_t>(int64_t) const;
template int64_t &ndarray<int64_t>::At<int64_t, int64_t>(int64_t, int64_t);
template const int64_t &ndarray<int64_t>::At<int64_t, int64_t>(int64_t,
                                                               int64_t) const;

template std::complex<float> &ndarray<std::complex<float>>::At();
template const std::complex<float> &ndarray<std::complex<float>>::At() const;
template std::complex<float> &
ndarray<std::complex<float>>::At<int64_t>(int64_t);
template const std::complex<float> &
ndarray<std::complex<float>>::At<int64_t>(int64_t) const;
template std::complex<float> &
ndarray<std::complex<float>>::At<int64_t, int64_t>(int64_t, int64_t);
template const std::complex<float> &
ndarray<std::complex<float>>::At<int64_t, int64_t>(int64_t, int64_t) const;

template std::complex<double> &ndarray<std::complex<double>>::At();
template const std::complex<double> &ndarray<std::complex<double>>::At() const;
template std::complex<double> &
ndarray<std::complex<double>>::At<int64_t>(int64_t);
template const std::complex<double> &
ndarray<std::complex<double>>::At<int64_t>(int64_t) const;
template std::complex<double> &
ndarray<std::complex<double>>::At<int64_t, int64_t>(int64_t, int64_t);
template const std::complex<double> &
ndarray<std::complex<double>>::At<int64_t, int64_t>(int64_t, int64_t) const;

template bool &ndarray<bool>::At();
template const bool &ndarray<bool>::At() const;
template bool &ndarray<bool>::At<int64_t>(int64_t);
template const bool &ndarray<bool>::At<int64_t>(int64_t) const;
template bool &ndarray<bool>::At<int64_t, int64_t>(int64_t, int64_t);
template const bool &ndarray<bool>::At<int64_t, int64_t>(int64_t,
                                                         int64_t) const;

} // namespace ndarray
