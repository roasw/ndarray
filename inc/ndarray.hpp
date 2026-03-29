#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <ATen/dlpack.h>
#include <armadillo>

namespace ndarray {

enum class Device { CPU, GPU };

template <typename T> class ndarray {
  private:
    DLManagedTensor *m_tensor;

  public:
    ndarray();
    ndarray(std::vector<int64_t> shape);
    ~ndarray();

    ndarray(const ndarray &) = delete;
    ndarray &operator=(const ndarray &) = delete;
    ndarray(ndarray &&other) noexcept;
    ndarray &operator=(ndarray &&other) noexcept;

    int64_t GetNdim() const;
    std::vector<int64_t> GetShape() const;
    T *GetData() const;

    template <typename... Indices> T &At(Indices... indices);

    template <typename... Indices> const T &At(Indices... indices) const;

    ndarray<T> Transpose() const;

    arma::Mat<T> &AsArmadillo();
    const arma::Mat<T> &AsArmadillo() const;

    DLManagedTensor *ToDLPack() const;

    ndarray<T> Add(const ndarray<T> &other) const;
    ndarray<T> Subtract(const ndarray<T> &other) const;
    ndarray<T> Multiply(const ndarray<T> &other) const;
    ndarray<T> Divide(const ndarray<T> &other) const;

    ndarray<T> operator+(const ndarray<T> &other) const;
    ndarray<T> operator-(const ndarray<T> &other) const;
    ndarray<T> operator*(const ndarray<T> &other) const;
    ndarray<T> operator/(const ndarray<T> &other) const;

  private:
    void AllocateTensor(const std::vector<int64_t> &shape);
    void DeallocateTensor();
};

} // namespace ndarray

#endif // NDARRAY_HPP
