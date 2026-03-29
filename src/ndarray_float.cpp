#include "ndarray.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ATen/dlpack.h>
#include <armadillo>
#include <c10/core/impl/alloc_cpu.h>

namespace ndarray {

namespace {
constexpr DLDataType GetDLDataType() { return {kDLFloat, 32, 1}; }
} // namespace

template <>
void ndarray<float>::AllocateTensor(const std::vector<int64_t> &shape) {
    m_tensor = new DLManagedTensor();

    m_tensor->dl_tensor.ndim = static_cast<int>(shape.size());

    int64_t *shape_ptr = new int64_t[shape.size()];
    int64_t *strides_ptr = new int64_t[shape.size()];

    for (size_t i = 0; i < shape.size(); ++i) {
        shape_ptr[i] = shape[i];
    }

    strides_ptr[0] = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        strides_ptr[i] = strides_ptr[i - 1] * shape[i - 1];
    }

    m_tensor->dl_tensor.shape = shape_ptr;
    m_tensor->dl_tensor.strides = strides_ptr;

    int64_t total_size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        total_size *= shape[i];
    }

    m_tensor->dl_tensor.data = c10::alloc_cpu(total_size * sizeof(float));
    m_tensor->dl_tensor.device = {kDLCPU, 0};
    m_tensor->dl_tensor.dtype = GetDLDataType();
    m_tensor->dl_tensor.byte_offset = 0;
    m_tensor->manager_ctx = nullptr;
    m_tensor->deleter = nullptr;
}

template <> void ndarray<float>::DeallocateTensor() {
    if (m_tensor) {
        if (m_tensor->dl_tensor.data) {
            c10::free_cpu(m_tensor->dl_tensor.data);
        }
        delete[] m_tensor->dl_tensor.shape;
        delete[] m_tensor->dl_tensor.strides;
        delete m_tensor;
        m_tensor = nullptr;
    }
}

template <> ndarray<float>::ndarray() : m_tensor(nullptr) {}

template <>
ndarray<float>::ndarray(std::vector<int64_t> shape) : m_tensor(nullptr) {
    AllocateTensor(shape);
}

template <> ndarray<float>::~ndarray() { DeallocateTensor(); }

template <>
ndarray<float>::ndarray(ndarray<float> &&other) noexcept
    : m_tensor(other.m_tensor) {
    other.m_tensor = nullptr;
}

template <>
ndarray<float> &ndarray<float>::operator=(ndarray<float> &&other) noexcept {
    if (this != &other) {
        DeallocateTensor();
        m_tensor = other.m_tensor;
        other.m_tensor = nullptr;
    }
    return *this;
}

template <> int64_t ndarray<float>::GetNdim() const {
    return m_tensor ? m_tensor->dl_tensor.ndim : 0;
}

template <> std::vector<int64_t> ndarray<float>::GetShape() const {
    if (!m_tensor) {
        return {};
    }
    return std::vector<int64_t>(m_tensor->dl_tensor.shape,
                                m_tensor->dl_tensor.shape +
                                    m_tensor->dl_tensor.ndim);
}

template <> float *ndarray<float>::GetData() const {
    if (!m_tensor) {
        return nullptr;
    }
    return static_cast<float *>(m_tensor->dl_tensor.data);
}

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
    int64_t idx = i0 * m_tensor->dl_tensor.strides[0];
    return static_cast<float *>(m_tensor->dl_tensor.data)[idx];
}

template <> template <> const float &ndarray<float>::At(int64_t i0) const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 1) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    int64_t idx = i0 * m_tensor->dl_tensor.strides[0];
    return static_cast<const float *>(m_tensor->dl_tensor.data)[idx];
}

template <> template <> float &ndarray<float>::At(int64_t i0, int64_t i1) {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    int64_t idx = i0 * m_tensor->dl_tensor.strides[0] +
                  i1 * m_tensor->dl_tensor.strides[1];
    return static_cast<float *>(m_tensor->dl_tensor.data)[idx];
}

template <>
template <>
const float &ndarray<float>::At(int64_t i0, int64_t i1) const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Invalid dimensions for At()");
    }
    int64_t idx = i0 * m_tensor->dl_tensor.strides[0] +
                  i1 * m_tensor->dl_tensor.strides[1];
    return static_cast<const float *>(m_tensor->dl_tensor.data)[idx];
}

template <> ndarray<float> ndarray<float>::Transpose() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("Transpose only supported for 2D arrays");
    }

    std::vector<int64_t> new_shape = {m_tensor->dl_tensor.shape[1],
                                      m_tensor->dl_tensor.shape[0]};
    ndarray<float> result(new_shape);

    int64_t *shape_ptr =
        const_cast<int64_t *>(result.m_tensor->dl_tensor.shape);
    int64_t *strides_ptr =
        const_cast<int64_t *>(result.m_tensor->dl_tensor.strides);

    shape_ptr[0] = m_tensor->dl_tensor.shape[1];
    shape_ptr[1] = m_tensor->dl_tensor.shape[0];
    strides_ptr[0] = m_tensor->dl_tensor.strides[1];
    strides_ptr[1] = m_tensor->dl_tensor.strides[0];

    c10::free_cpu(result.m_tensor->dl_tensor.data);
    result.m_tensor->dl_tensor.data = m_tensor->dl_tensor.data;

    return result;
}

template <> arma::Mat<float> &ndarray<float>::AsArmadillo() {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("AsArmadillo only supported for 2D arrays");
    }
    arma::Mat<float> *mat =
        new arma::Mat<float>(static_cast<float *>(m_tensor->dl_tensor.data),
                             m_tensor->dl_tensor.shape[0],
                             m_tensor->dl_tensor.shape[1], false, true);
    return *mat;
}

template <> const arma::Mat<float> &ndarray<float>::AsArmadillo() const {
    if (!m_tensor || m_tensor->dl_tensor.ndim != 2) {
        throw std::runtime_error("AsArmadillo only supported for 2D arrays");
    }
    arma::Mat<float> *mat =
        new arma::Mat<float>(static_cast<float *>(m_tensor->dl_tensor.data),
                             m_tensor->dl_tensor.shape[0],
                             m_tensor->dl_tensor.shape[1], false, true);
    return *const_cast<const arma::Mat<float> *>(mat);
}

template <> DLManagedTensor *ndarray<float>::ToDLPack() const {
    return m_tensor;
}

template <>
ndarray<float> ndarray<float>::Add(const ndarray<float> &other) const {
    if (!m_tensor || !other.m_tensor) {
        throw std::runtime_error("Cannot add empty arrays");
    }
    arma::Mat<float> this_mat(GetData(), m_tensor->dl_tensor.shape[0],
                              m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> other_mat(other.GetData(),
                               other.m_tensor->dl_tensor.shape[0],
                               other.m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> result = this_mat + other_mat;
    ndarray<float> out(GetShape());
    std::copy(result.memptr(), result.memptr() + result.n_elem, out.GetData());
    return out;
}

template <>
ndarray<float> ndarray<float>::Subtract(const ndarray<float> &other) const {
    if (!m_tensor || !other.m_tensor) {
        throw std::runtime_error("Cannot subtract empty arrays");
    }
    arma::Mat<float> this_mat(GetData(), m_tensor->dl_tensor.shape[0],
                              m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> other_mat(other.GetData(),
                               other.m_tensor->dl_tensor.shape[0],
                               other.m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> result = this_mat - other_mat;
    ndarray<float> out(GetShape());
    std::copy(result.memptr(), result.memptr() + result.n_elem, out.GetData());
    return out;
}

template <>
ndarray<float> ndarray<float>::Multiply(const ndarray<float> &other) const {
    if (!m_tensor || !other.m_tensor) {
        throw std::runtime_error("Cannot multiply empty arrays");
    }
    arma::Mat<float> this_mat(GetData(), m_tensor->dl_tensor.shape[0],
                              m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> other_mat(other.GetData(),
                               other.m_tensor->dl_tensor.shape[0],
                               other.m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> result = this_mat % other_mat;
    ndarray<float> out(GetShape());
    std::copy(result.memptr(), result.memptr() + result.n_elem, out.GetData());
    return out;
}

template <>
ndarray<float> ndarray<float>::Divide(const ndarray<float> &other) const {
    if (!m_tensor || !other.m_tensor) {
        throw std::runtime_error("Cannot divide empty arrays");
    }
    arma::Mat<float> this_mat(GetData(), m_tensor->dl_tensor.shape[0],
                              m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> other_mat(other.GetData(),
                               other.m_tensor->dl_tensor.shape[0],
                               other.m_tensor->dl_tensor.shape[1], false, true);
    arma::Mat<float> result = this_mat / other_mat;
    ndarray<float> out(GetShape());
    std::copy(result.memptr(), result.memptr() + result.n_elem, out.GetData());
    return out;
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
