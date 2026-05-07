#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <ATen/dlpack.h>
#include <armadillo>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

namespace ndarray {

// Lifetime-extending Armadillo view.
// Keeps the underlying DLManagedTensor alive for as long as this view exists.
// Implicitly converts to arma::Mat<T>& so existing Armadillo code works with
// minimal changes — store this type, not arma::Mat<T>, to avoid dangling refs.
template <typename T> struct ArmadilloView {
    arma::Mat<T> mat; // non-owning view into DLManagedTensor data
    std::shared_ptr<DLManagedTensor>
        owner; // ref-count bump — keeps tensor alive

    operator arma::Mat<T> &() { return mat; }
    operator const arma::Mat<T> &() const { return mat; }
};

template <typename T> class ndarray {
  private:
    std::shared_ptr<DLManagedTensor> m_tensor;

  public:
    ndarray();
    ndarray(std::vector<int64_t> shape,
            c10::DeviceType device = c10::DeviceType::CPU);

    // Value semantics: copy is zero-copy (shared ownership of DLManagedTensor)
    ndarray(const ndarray &) = default;
    ndarray &operator=(const ndarray &) = default;
    ndarray(ndarray &&) noexcept = default;
    ndarray &operator=(ndarray &&) noexcept = default;
    ~ndarray() = default;

    int64_t GetNdim() const;
    std::vector<int64_t> GetShape() const;
    std::vector<int64_t> GetStrides() const;
    c10::DeviceType GetDevice() const;
    T *GetData() const;

    template <typename... Indices> T &At(Indices... indices);
    template <typename... Indices> const T &At(Indices... indices) const;

    ndarray<T> Transpose() const;

    // Explicit deep copy
    ndarray<T> Clone() const;

    // Armadillo escape hatch — 2D only.
    // Returns a lifetime-extending view; store as ArmadilloView<T>,
    // not arma::Mat<T>, to avoid dangling references.
    ArmadilloView<T> AsArmadillo();
    ArmadilloView<T> AsArmadillo() const;

    // Zero-copy DLPack export.
    // manager_ctx holds a new shared_ptr<DLManagedTensor> bump;
    // buffer lives until consumer calls
    // managed_tensor->deleter(managed_tensor).
    DLManagedTensor *ToDLPack() const;

    // Takes ownership of a DLManagedTensor and wraps it in ndarray.
    static ndarray FromDLPack(DLManagedTensor *managed_tensor);

    ndarray<T> Add(const ndarray<T> &other) const;
    ndarray<T> Subtract(const ndarray<T> &other) const;
    ndarray<T> Multiply(const ndarray<T> &other) const;
    ndarray<T> Divide(const ndarray<T> &other) const;

    ndarray<T> operator+(const ndarray<T> &other) const;
    ndarray<T> operator-(const ndarray<T> &other) const;
    ndarray<T> operator*(const ndarray<T> &other) const;
    ndarray<T> operator/(const ndarray<T> &other) const;

  private:
    // Internal factory constructor used by Transpose(), Clone(), etc.
    explicit ndarray(std::shared_ptr<DLManagedTensor> tensor);
};

} // namespace ndarray

#endif // NDARRAY_HPP
