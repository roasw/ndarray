#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ATen/dlpack.h>
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

int64_t NumElements(const DLTensor &t) {
    int64_t n = 1;
    for (int i = 0; i < t.ndim; ++i) {
        n *= t.shape[i];
    }
    return n;
}

DLDevice ToDLDevice(c10::DeviceType device_type) {
    switch (device_type) {
    case c10::DeviceType::CPU:
        return {kDLCPU, 0};
    case c10::DeviceType::CUDA:
        return {kDLCUDA, 0};
    default:
        throw std::runtime_error("Unsupported device type");
    }
}

c10::DeviceType FromDLDevice(DLDeviceType device_type) {
    switch (device_type) {
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
AllocateTensor(const std::vector<int64_t> &shape, c10::DeviceType device_type) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ndarray only supports float and double");

    const int ndim = static_cast<int>(shape.size());
    const int64_t total = [&] {
        int64_t n = 1;
        for (auto s : shape) {
            n *= s;
        }
        return n;
    }();

    c10::Allocator *alloc = c10::GetAllocator(device_type);
    c10::DataPtr data_ptr =
        alloc->allocate(total * static_cast<int64_t>(sizeof(T)));

    auto *mt = new DLManagedTensor();
    mt->dl_tensor.data = data_ptr.get();
    mt->dl_tensor.device = ToDLDevice(device_type);
    mt->dl_tensor.ndim = ndim;
    mt->dl_tensor.dtype = DTypeTraits<T>::kDType();
    mt->dl_tensor.byte_offset = 0;

    int64_t *shape_ptr = nullptr;
    int64_t *strides_ptr = nullptr;

    if (ndim > 0) {
        shape_ptr = new int64_t[ndim];
        strides_ptr = new int64_t[ndim];

        for (int i = 0; i < ndim; ++i) {
            shape_ptr[i] = shape[i];
        }

        strides_ptr[0] = 1;
        for (int i = 1; i < ndim; ++i) {
            strides_ptr[i] = strides_ptr[i - 1] * shape[i - 1];
        }
    }

    mt->dl_tensor.shape = shape_ptr;
    mt->dl_tensor.strides = strides_ptr;
    mt->manager_ctx = new c10::DataPtr(std::move(data_ptr));
    mt->deleter = OwningDeleter;

    return {mt, OwningDeleter};
}

template <typename T> bool IsExpectedDType(const DLTensor &t) {
    const auto dtype = DTypeTraits<T>::kDType();
    return t.dtype.code == dtype.code && t.dtype.bits == dtype.bits &&
           t.dtype.lanes == dtype.lanes;
}

template <typename T, typename Op>
ndarray<T> ArmaOp(const ndarray<T> &a, const ndarray<T> &b, const char *op_name,
                  Op op) {
    if (!a.GetData() || !b.GetData()) {
        throw std::runtime_error(std::string("Cannot ") + op_name +
                                 " empty arrays");
    }
    const auto shape_a = a.GetShape();
    const auto shape_b = b.GetShape();
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        throw std::runtime_error(std::string(op_name) +
                                 " only supported for 2D arrays");
    }
    if (shape_a[0] != shape_b[0] || shape_a[1] != shape_b[1]) {
        throw std::runtime_error(std::string("Shape mismatch for ") + op_name);
    }

    arma::Mat<T> ma(a.GetData(), shape_a[0], shape_a[1], false, true);
    arma::Mat<T> mb(b.GetData(), shape_b[0], shape_b[1], false, true);
    arma::Mat<T> result = op(ma, mb);

    ndarray<T> out(a.GetShape(), a.GetDevice());
    std::copy_n(result.memptr(), static_cast<int64_t>(result.n_elem),
                out.GetData());
    return out;
}

} // namespace

template <typename T>
ndarray<T>::ndarray(std::shared_ptr<DLManagedTensor> tensor)
    : m_tensor(std::move(tensor)) {}

template <typename T> ndarray<T>::ndarray() : m_tensor(nullptr) {}

template <typename T>
ndarray<T>::ndarray(std::vector<int64_t> shape, c10::DeviceType device)
    : m_tensor(AllocateTensor<T>(shape, device)) {}

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
    static_assert((std::is_convertible_v<Indices, int64_t> && ...),
                  "Indices must be convertible to int64_t");

    if (!m_tensor) {
        throw std::runtime_error("ndarray is empty");
    }

    constexpr size_t kCount = sizeof...(Indices);
    const auto &t = m_tensor->dl_tensor;

    if constexpr (kCount == 0) {
        return *static_cast<T *>(t.data);
    } else if constexpr (kCount == 1) {
        if (t.ndim != 1) {
            throw std::runtime_error("Invalid dimensions for At()");
        }
        const std::array<int64_t, 1> idx = {static_cast<int64_t>(indices)...};
        return static_cast<T *>(t.data)[idx[0] * t.strides[0]];
    } else if constexpr (kCount == 2) {
        if (t.ndim != 2) {
            throw std::runtime_error("Invalid dimensions for At()");
        }
        const std::array<int64_t, 2> idx = {static_cast<int64_t>(indices)...};
        return static_cast<T *>(
            t.data)[idx[0] * t.strides[0] + idx[1] * t.strides[1]];
    } else {
        static_assert(kCount <= 2, "At() supports up to 2 indices");
    }
}

template <typename T>
template <typename... Indices>
const T &ndarray<T>::At(Indices... indices) const {
    static_assert((std::is_convertible_v<Indices, int64_t> && ...),
                  "Indices must be convertible to int64_t");

    if (!m_tensor) {
        throw std::runtime_error("ndarray is empty");
    }

    constexpr size_t kCount = sizeof...(Indices);
    const auto &t = m_tensor->dl_tensor;

    if constexpr (kCount == 0) {
        return *static_cast<const T *>(t.data);
    } else if constexpr (kCount == 1) {
        if (t.ndim != 1) {
            throw std::runtime_error("Invalid dimensions for At()");
        }
        const std::array<int64_t, 1> idx = {static_cast<int64_t>(indices)...};
        return static_cast<const T *>(t.data)[idx[0] * t.strides[0]];
    } else if constexpr (kCount == 2) {
        if (t.ndim != 2) {
            throw std::runtime_error("Invalid dimensions for At()");
        }
        const std::array<int64_t, 2> idx = {static_cast<int64_t>(indices)...};
        return static_cast<const T *>(
            t.data)[idx[0] * t.strides[0] + idx[1] * t.strides[1]];
    } else {
        static_assert(kCount <= 2, "At() supports up to 2 indices");
    }
}

template <typename T> ndarray<T> ndarray<T>::Transpose() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Transpose only supported for 2D arrays");
    }

    const auto &t = m_tensor->dl_tensor;
    std::vector<int64_t> new_shape = {t.shape[1], t.shape[0]};
    ndarray<T> result(new_shape, GetDevice());

    arma::Mat<T> src(GetData(), t.shape[0], t.shape[1], false, true);
    arma::Mat<T> dst(result.GetData(), result.m_tensor->dl_tensor.shape[0],
                     result.m_tensor->dl_tensor.shape[1], false, true);
    dst = src.t();
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

template <typename T> ArmadilloView<T> ndarray<T>::AsArmadillo() {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("AsArmadillo only supported for 2D arrays");
    }

    const auto &t = m_tensor->dl_tensor;
    return {arma::Mat<T>(static_cast<T *>(t.data), t.shape[0], t.shape[1],
                         false, true),
            m_tensor};
}

template <typename T> ArmadilloView<T> ndarray<T>::AsArmadillo() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("AsArmadillo only supported for 2D arrays");
    }

    const auto &t = m_tensor->dl_tensor;
    return {arma::Mat<T>(static_cast<T *>(t.data), t.shape[0], t.shape[1],
                         false, true),
            m_tensor};
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

    int64_t *shape_ptr = nullptr;
    int64_t *strides_ptr = nullptr;

    if (src.ndim > 0) {
        shape_ptr = new int64_t[src.ndim];
        strides_ptr = new int64_t[src.ndim];
        std::copy_n(src.shape, src.ndim, shape_ptr);
        std::copy_n(src.strides, src.ndim, strides_ptr);
    }

    mt->dl_tensor.shape = shape_ptr;
    mt->dl_tensor.strides = strides_ptr;
    mt->manager_ctx = new std::shared_ptr<DLManagedTensor>(m_tensor);
    mt->deleter = ViewDeleter;

    return mt;
}

template <typename T>
ndarray<T> ndarray<T>::FromDLPack(DLManagedTensor *managed_tensor) {
    if (!managed_tensor) {
        return ndarray<T>();
    }

    const auto &t = managed_tensor->dl_tensor;
    if (!IsExpectedDType<T>(t)) {
        ManagedTensorDeleter(managed_tensor);
        throw std::runtime_error("FromDLPack expects matching floating dtype");
    }

    return ndarray<T>(
        std::shared_ptr<DLManagedTensor>(managed_tensor, ManagedTensorDeleter));
}

template <typename T>
ndarray<T> ndarray<T>::Add(const ndarray<T> &other) const {
    return ArmaOp(*this, other, "Add",
                  [](const arma::Mat<T> &a, const arma::Mat<T> &b) {
                      return arma::Mat<T>(a + b);
                  });
}

template <typename T>
ndarray<T> ndarray<T>::Subtract(const ndarray<T> &other) const {
    return ArmaOp(*this, other, "Subtract",
                  [](const arma::Mat<T> &a, const arma::Mat<T> &b) {
                      return arma::Mat<T>(a - b);
                  });
}

template <typename T>
ndarray<T> ndarray<T>::Multiply(const ndarray<T> &other) const {
    return ArmaOp(*this, other, "Multiply",
                  [](const arma::Mat<T> &a, const arma::Mat<T> &b) {
                      return arma::Mat<T>(a % b);
                  });
}

template <typename T>
ndarray<T> ndarray<T>::Divide(const ndarray<T> &other) const {
    return ArmaOp(*this, other, "Divide",
                  [](const arma::Mat<T> &a, const arma::Mat<T> &b) {
                      return arma::Mat<T>(a / b);
                  });
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

} // namespace ndarray
