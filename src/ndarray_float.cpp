#include "ndarray.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ATen/dlpack.h>
#include <armadillo>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

namespace ndarray {

namespace {

constexpr DLDataType kFloatDType() { return {kDLFloat, 32, 1}; }

int64_t NumElements(const DLTensor &t) {
    int64_t n = 1;
    for (int i = 0; i < t.ndim; ++i) {
        n *= t.shape[i];
    }
    return n;
}

// Converts c10::DeviceType to DLDevice.
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

// Converts DLDeviceType to c10::DeviceType.
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

// Deleter for the owning DLManagedTensor (stored in
// shared_ptr<DLManagedTensor>). manager_ctx holds a heap-allocated c10::DataPtr
// whose destructor returns memory to the correct device pool. Shape and strides
// are freed here too.
void OwningDeleter(DLManagedTensor *mt) {
    if (!mt) {
        return;
    }
    delete[] mt->dl_tensor.shape;
    delete[] mt->dl_tensor.strides;
    // Return data to the c10 pool by destroying the DataPtr.
    delete static_cast<c10::DataPtr *>(mt->manager_ctx);
    delete mt;
}

// Deleter for the non-owning DLManagedTensor handed out by ToDLPack().
// manager_ctx holds a heap-allocated shared_ptr<DLManagedTensor> that was
// bumped in ToDLPack(); destroying it decrements the refcount.
void ViewDeleter(DLManagedTensor *mt) {
    if (!mt) {
        return;
    }
    delete[] mt->dl_tensor.shape;
    delete[] mt->dl_tensor.strides;
    delete static_cast<std::shared_ptr<DLManagedTensor> *>(mt->manager_ctx);
    delete mt;
}

// Allocates a fresh owning DLManagedTensor for the given shape and device.
// Column-major (Fortran) strides are set for Armadillo compatibility.
std::shared_ptr<DLManagedTensor>
AllocateTensor(const std::vector<int64_t> &shape, c10::DeviceType device_type) {
    const int ndim = static_cast<int>(shape.size());
    const int64_t total = [&] {
        int64_t n = 1;
        for (auto s : shape)
            n *= s;
        return n;
    }();

    c10::Allocator *alloc = c10::GetAllocator(device_type);
    c10::DataPtr data_ptr = alloc->allocate(total * sizeof(float));

    auto *mt = new DLManagedTensor();
    mt->dl_tensor.data = data_ptr.get();
    mt->dl_tensor.device = ToDLDevice(device_type);
    mt->dl_tensor.ndim = ndim;
    mt->dl_tensor.dtype = kFloatDType();
    mt->dl_tensor.byte_offset = 0;

    int64_t *shape_ptr = nullptr;
    int64_t *strides_ptr = nullptr;

    if (ndim > 0) {
        shape_ptr = new int64_t[ndim];
        strides_ptr = new int64_t[ndim];

        for (int i = 0; i < ndim; ++i) {
            shape_ptr[i] = shape[i];
        }

        // Column-major (Fortran) strides.
        strides_ptr[0] = 1;
        for (int i = 1; i < ndim; ++i) {
            strides_ptr[i] = strides_ptr[i - 1] * shape[i - 1];
        }
    }

    mt->dl_tensor.shape = shape_ptr;
    mt->dl_tensor.strides = strides_ptr;

    // Transfer DataPtr ownership into manager_ctx so OwningDeleter can return
    // memory to the correct c10 pool on destruction.
    mt->manager_ctx = new c10::DataPtr(std::move(data_ptr));
    mt->deleter = OwningDeleter;

    return {mt, OwningDeleter};
}

} // namespace

// ---------------------------------------------------------------------------
// Private factory constructor
// ---------------------------------------------------------------------------

template <>
ndarray<float>::ndarray(std::shared_ptr<DLManagedTensor> tensor)
    : m_tensor(std::move(tensor)) {}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <> ndarray<float>::ndarray() : m_tensor(nullptr) {}

template <>
ndarray<float>::ndarray(std::vector<int64_t> shape, c10::DeviceType device)
    : m_tensor(AllocateTensor(shape, device)) {}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <> int64_t ndarray<float>::GetNdim() const {
    return m_tensor ? m_tensor->dl_tensor.ndim : 0;
}

template <> std::vector<int64_t> ndarray<float>::GetShape() const {
    if (!m_tensor) {
        return {};
    }
    const auto &t = m_tensor->dl_tensor;
    return {t.shape, t.shape + t.ndim};
}

template <> std::vector<int64_t> ndarray<float>::GetStrides() const {
    if (!m_tensor) {
        return {};
    }
    const auto &t = m_tensor->dl_tensor;
    return {t.strides, t.strides + t.ndim};
}

template <> c10::DeviceType ndarray<float>::GetDevice() const {
    if (!m_tensor) {
        return c10::DeviceType::CPU;
    }
    return FromDLDevice(m_tensor->dl_tensor.device.device_type);
}

template <> float *ndarray<float>::GetData() const {
    if (!m_tensor) {
        return nullptr;
    }
    return static_cast<float *>(m_tensor->dl_tensor.data);
}

// ---------------------------------------------------------------------------
// At() — explicit specializations for 0D, 1D, 2D
// ---------------------------------------------------------------------------

template <> template <> float &ndarray<float>::At() {
    if (!m_tensor) {
        throw std::runtime_error("ndarray is empty");
    }
    return *static_cast<float *>(m_tensor->dl_tensor.data);
}

template <> template <> const float &ndarray<float>::At() const {
    if (!m_tensor) {
        throw std::runtime_error("ndarray is empty");
    }
    return *static_cast<const float *>(m_tensor->dl_tensor.data);
}

template <> template <> float &ndarray<float>::At(int64_t i0) {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 1) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    return static_cast<float *>(
        m_tensor->dl_tensor.data)[i0 * m_tensor->dl_tensor.strides[0]];
}

template <> template <> const float &ndarray<float>::At(int64_t i0) const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 1) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    return static_cast<const float *>(
        m_tensor->dl_tensor.data)[i0 * m_tensor->dl_tensor.strides[0]];
}

template <> template <> float &ndarray<float>::At(int64_t i0, int64_t i1) {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    const auto &t = m_tensor->dl_tensor;
    return static_cast<float *>(t.data)[i0 * t.strides[0] + i1 * t.strides[1]];
}

template <>
template <>
const float &ndarray<float>::At(int64_t i0, int64_t i1) const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    const auto &t = m_tensor->dl_tensor;
    return static_cast<const float *>(
        t.data)[i0 * t.strides[0] + i1 * t.strides[1]];
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

template <> ndarray<float> ndarray<float>::Transpose() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Transpose only supported for 2D arrays");
    }
    const auto &t = m_tensor->dl_tensor;
    std::vector<int64_t> new_shape = {t.shape[1], t.shape[0]};
    ndarray<float> result(new_shape, GetDevice());

    arma::Mat<float> src(GetData(), t.shape[0], t.shape[1], false, true);
    arma::Mat<float> dst(result.GetData(), result.m_tensor->dl_tensor.shape[0],
                         result.m_tensor->dl_tensor.shape[1], false, true);
    dst = src.t();

    return result;
}

// ---------------------------------------------------------------------------
// Clone — explicit deep copy
// ---------------------------------------------------------------------------

template <> ndarray<float> ndarray<float>::Clone() const {
    if (!m_tensor) {
        return ndarray<float>();
    }
    const auto &t = m_tensor->dl_tensor;
    ndarray<float> result(GetShape(), GetDevice());
    std::copy_n(static_cast<const float *>(t.data), NumElements(t),
                result.GetData());
    return result;
}

// ---------------------------------------------------------------------------
// AsArmadillo — lifetime-extending Armadillo view (2D only)
// ---------------------------------------------------------------------------

template <> ArmadilloView<float> ndarray<float>::AsArmadillo() {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("AsArmadillo only supported for 2D arrays");
    }
    const auto &t = m_tensor->dl_tensor;
    return {arma::Mat<float>(static_cast<float *>(t.data), t.shape[0],
                             t.shape[1], false, true),
            m_tensor};
}

template <> ArmadilloView<float> ndarray<float>::AsArmadillo() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("AsArmadillo only supported for 2D arrays");
    }
    const auto &t = m_tensor->dl_tensor;
    return {arma::Mat<float>(static_cast<float *>(t.data), t.shape[0],
                             t.shape[1], false, true),
            m_tensor};
}

// ---------------------------------------------------------------------------
// ToDLPack — zero-copy export
// ---------------------------------------------------------------------------

template <> DLManagedTensor *ndarray<float>::ToDLPack() const {
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

    // Bump the shared_ptr refcount so the owning tensor stays alive until
    // the consumer calls mt->deleter(mt).
    mt->manager_ctx = new std::shared_ptr<DLManagedTensor>(m_tensor);
    mt->deleter = ViewDeleter;

    return mt;
}

// ---------------------------------------------------------------------------
// Arithmetic — internal Armadillo helpers
// ---------------------------------------------------------------------------

namespace {

ndarray<float> ArmaOp(const ndarray<float> &a, const ndarray<float> &b,
                      const char *op_name,
                      arma::Mat<float> (*op)(const arma::Mat<float> &,
                                             const arma::Mat<float> &)) {
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
    arma::Mat<float> ma(a.GetData(), shape_a[0], shape_a[1], false, true);
    arma::Mat<float> mb(b.GetData(), shape_b[0], shape_b[1], false, true);
    arma::Mat<float> result = op(ma, mb);
    ndarray<float> out(a.GetShape(), a.GetDevice());
    std::copy_n(result.memptr(), result.n_elem, out.GetData());
    return out;
}

} // namespace

template <>
ndarray<float> ndarray<float>::Add(const ndarray<float> &other) const {
    return ArmaOp(*this, other, "Add",
                  [](const arma::Mat<float> &a, const arma::Mat<float> &b) {
                      return arma::Mat<float>(a + b);
                  });
}

template <>
ndarray<float> ndarray<float>::Subtract(const ndarray<float> &other) const {
    return ArmaOp(*this, other, "Subtract",
                  [](const arma::Mat<float> &a, const arma::Mat<float> &b) {
                      return arma::Mat<float>(a - b);
                  });
}

template <>
ndarray<float> ndarray<float>::Multiply(const ndarray<float> &other) const {
    return ArmaOp(*this, other, "Multiply",
                  [](const arma::Mat<float> &a, const arma::Mat<float> &b) {
                      return arma::Mat<float>(a % b);
                  });
}

template <>
ndarray<float> ndarray<float>::Divide(const ndarray<float> &other) const {
    return ArmaOp(*this, other, "Divide",
                  [](const arma::Mat<float> &a, const arma::Mat<float> &b) {
                      return arma::Mat<float>(a / b);
                  });
}

template <>
ndarray<float> ndarray<float>::operator+(const ndarray<float> &other) const {
    return Add(other);
}

template <>
ndarray<float> ndarray<float>::operator-(const ndarray<float> &other) const {
    return Subtract(other);
}

template <>
ndarray<float> ndarray<float>::operator*(const ndarray<float> &other) const {
    return Multiply(other);
}

template <>
ndarray<float> ndarray<float>::operator/(const ndarray<float> &other) const {
    return Divide(other);
}

} // namespace ndarray
