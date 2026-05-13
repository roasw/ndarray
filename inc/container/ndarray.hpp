#pragma once

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

/**
 * @brief Lifetime-extending Armadillo view for 2D ndarray storage.
 *
 * @details
 * `ndarray<T>::ToArmadilloView()` (implemented in
 * `src/container/ndarray_typed.cpp`) constructs this object by:
 * - creating a non-owning `arma::Mat<T>` over the existing tensor buffer,
 * - storing a `std::shared_ptr<DLManagedTensor>` owner to keep storage alive.
 *
 * This makes Armadillo interop zero-copy while preserving lifetime safety.
 *
 * @note Keep this wrapper object alive. If only `arma::Mat<T>` is kept, the
 * underlying storage can outlive incorrectly and become dangling.
 */
template <typename T> struct ArmadilloView {
    /// Non-owning Armadillo view into ndarray memory.
    arma::Mat<T> mat;

    /// Shared ownership bump that keeps the tensor storage alive.
    std::shared_ptr<DLManagedTensor> owner;

    /// Convenience conversion for APIs expecting mutable `arma::Mat<T>&`.
    operator arma::Mat<T> &() { return mat; }

    /// Convenience conversion for APIs expecting const `arma::Mat<T>&`.
    operator const arma::Mat<T> &() const { return mat; }
};

/**
 * @brief Bool specialization does not expose an Armadillo matrix.
 *
 * @details Armadillo does not support `arma::Mat<bool>`. The bool
 * `ToArmadilloView()` path in `src/container/ndarray_typed.cpp` throws, so this
 * specialization only carries the owner type used by declarations.
 */
template <> struct ArmadilloView<bool> {
    /// Present for type completeness; bool Armadillo views are unsupported.
    std::shared_ptr<DLManagedTensor> owner;
};

/**
 * @brief Column-major tensor container backed by `DLManagedTensor`.
 *
 * @details
 * The implementation lives in `src/container/ndarray_typed.cpp` and fulfills
 * these guarantees:
 * - storage and metadata are DLPack-compatible,
 * - copies are zero-copy shared ownership,
 * - explicit clone is deep copy,
 * - DLPack bridge preserves ownership and zero-copy semantics.
 *
 * Allocation uses `c10::GetAllocator(device)` and creates column-major strides
 * (`stride[0] = 1`, `stride[i] = stride[i - 1] * shape[i - 1]`).
 */
template <typename T> class ndarray {
  private:
    std::shared_ptr<DLManagedTensor> m_tensor;

  public:
    /**
     * @brief Create an empty ndarray.
     *
     * @details Implemented as `m_tensor == nullptr` in
     * `src/container/ndarray_typed.cpp`.
     */
    ndarray();

    /**
     * @brief Allocate a new ndarray for shape/device.
     * @param shape Logical tensor shape.
     * @param deviceType Target device (CPU or CUDA in current implementation).
     *
     * @details
     * Delegates to `AllocateTensor<T>()` in `src/container/ndarray_typed.cpp`,
     * which allocates data via c10 allocators and populates DLPack metadata.
     *
     * @note Memory layout is column-major by construction.
     */
    ndarray(std::vector<int64_t> shape,
            c10::DeviceType deviceType = c10::DeviceType::CPU);

    /**
     * @brief Copy constructor with zero-copy semantics.
     *
     * @details
     * This constructor is defaulted and the only data member is
     * `std::shared_ptr<DLManagedTensor> m_tensor`. Therefore, copying an
     * `ndarray<T>` copies only the shared pointer control block/reference, not
     * the underlying tensor buffer.
     *
     * In practice this means both arrays alias the same storage and observe
     * each other's writes, while ownership remains reference-counted.
     */
    ndarray(const ndarray &) = default;

    /**
     * @brief Copy assignment with zero-copy semantics.
     *
     * @details Same reasoning as the copy constructor: defaulted assignment
     * assigns `m_tensor` by shared-pointer semantics, so storage is shared and
     * not duplicated.
     */
    ndarray &operator=(const ndarray &) = default;

    /**
     * @brief Move constructor.
     *
     * @details Transfers shared ownership handle without copying tensor data.
     */
    ndarray(ndarray &&) noexcept = default;

    /**
     * @brief Move assignment.
     *
     * @details Transfers shared ownership handle without copying tensor data.
     */
    ndarray &operator=(ndarray &&) noexcept = default;

    /**
     * @brief Destructor.
     *
     * @details Lifetime is managed by `std::shared_ptr` deleters defined in
     * `src/container/ndarray_typed.cpp`.
     */
    ~ndarray() = default;

    /**
     * @brief Return number of dimensions.
     * @return Tensor rank, or 0 for empty ndarray.
     */
    int64_t GetNdim() const;

    /**
     * @brief Return shape vector.
     * @return Empty vector for empty ndarray.
     */
    std::vector<int64_t> GetShape() const;

    /**
     * @brief Return stride vector in element units.
     * @return Empty vector for empty ndarray.
     */
    std::vector<int64_t> GetStrides() const;

    /**
     * @brief Return c10 device type inferred from DLPack device.
     * @return CPU for empty ndarray; otherwise mapped device.
     */
    c10::DeviceType GetDevice() const;

    /**
     * @brief Return raw data pointer.
     * @return `nullptr` for empty ndarray.
     */
    T *GetData() const;

    /**
     * @brief Mutable element access.
     * @tparam Indices Zero, one, or two indices convertible to `int64_t`.
     * @return Reference to selected element.
     *
     * @details
     * Implementation in `src/container/ndarray_typed.cpp` validates empty
     * state, rank/index arity compatibility, and bounds before computing
     * offset using stored strides.
     */
    template <typename... Indices> T &At(Indices... indices);

    /**
     * @brief Const element access.
     * @tparam Indices Zero, one, or two indices convertible to `int64_t`.
     * @return Const reference to selected element.
     *
     * @details Same checks and indexing path as mutable overload.
     */
    template <typename... Indices> const T &At(Indices... indices) const;

    /**
     * @brief Return transposed 2D array.
     * @return New ndarray with swapped shape.
     *
     * @details
     * Implemented in `src/container/ndarray_typed.cpp`:
     * - non-bool types use Armadillo transpose on zero-copy matrix views,
     * - bool uses explicit element-wise transpose because Armadillo bool
     *   matrices are unsupported.
     */
    ndarray<T> Transpose() const;

    /**
     * @brief Deep-copy tensor storage.
     * @return New ndarray with independent storage and copied contents.
     *
     * @details `Clone()` allocates a fresh tensor and `std::copy_n` transfers
     * all elements in `src/container/ndarray_typed.cpp`.
     */
    ndarray<T> Clone() const;

    /**
     * @brief Build lifetime-safe Armadillo view for mutable access.
     * @return `ArmadilloView<T>` wrapping non-owning `arma::Mat<T>` plus owner.
     *
     * @details
     * Implemented in `src/container/ndarray_typed.cpp` as a zero-copy view over
     * existing data. Requires 2D shape.
     *
     * @throws std::runtime_error if ndarray is empty, not 2D, or `T=bool`.
     */
    ArmadilloView<T> ToArmadilloView();

    /**
     * @brief Build lifetime-safe Armadillo view for const access.
     * @return `ArmadilloView<T>` wrapping non-owning `arma::Mat<T>` plus owner.
     *
     * @details Same behavior and constraints as non-const overload.
     */
    ArmadilloView<T> ToArmadilloView() const;

    /**
     * @brief Export a DLPack view without copying data.
     * @return Newly allocated `DLManagedTensor*` view, or `nullptr` if empty.
     *
     * @details
     * In `src/container/ndarray_typed.cpp`, this creates a new managed wrapper
     * with copied shape/stride arrays and a `manager_ctx` that owns a
     * `shared_ptr<DLManagedTensor>` bump, so underlying storage stays alive
     * until the consumer calls the DLPack deleter.
     */
    DLManagedTensor *ToDLPack() const;

    /**
     * @brief Adopt ownership of a producer-provided DLPack tensor.
     * @param managedTensor Pointer transferred to ndarray ownership.
     * @return ndarray sharing the provided storage.
     *
     * @details
     * `FromDLPack()` in `src/container/ndarray_typed.cpp` validates dtype
     * compatibility (`code/bits/lanes`) and wraps the pointer in
     * `std::shared_ptr` with `ManagedTensorDeleter`.
     *
     * @throws std::runtime_error on dtype mismatch.
     */
    static ndarray FromDLPack(DLManagedTensor *managedTensor);

    /**
     * @brief Element-wise add for 2D arrays.
     *
     * @details
     * Implemented in `src/container/ndarray_typed.cpp`:
     * - non-bool: torch-backed element-wise add,
     * - bool: explicit logical-OR path.
     */
    ndarray<T> Add(const ndarray<T> &other) const;

    /**
     * @brief Element-wise subtract for 2D arrays.
     *
     * @details
     * Implemented in `src/container/ndarray_typed.cpp`:
     * - non-bool: torch-backed element-wise subtract,
     * - bool: explicitly unsupported and throws.
     */
    ndarray<T> Subtract(const ndarray<T> &other) const;

    /**
     * @brief Element-wise multiply for 2D arrays.
     *
     * @details
     * Implemented in `src/container/ndarray_typed.cpp`:
     * - non-bool: torch-backed element-wise multiply,
     * - bool: explicit logical-AND path.
     */
    ndarray<T> Multiply(const ndarray<T> &other) const;

    /**
     * @brief Element-wise divide for 2D arrays.
     *
     * @details
     * Implemented in `src/container/ndarray_typed.cpp`:
     * - non-bool: torch-backed element-wise division,
     * - bool: explicitly unsupported and throws.
     */
    ndarray<T> Divide(const ndarray<T> &other) const;

    /** @brief Convenience alias for `Add(other)`. */
    ndarray<T> operator+(const ndarray<T> &other) const;

    /** @brief Convenience alias for `Subtract(other)`. */
    ndarray<T> operator-(const ndarray<T> &other) const;

    /** @brief Convenience alias for `Multiply(other)`. */
    ndarray<T> operator*(const ndarray<T> &other) const;

    /** @brief Convenience alias for `Divide(other)`. */
    ndarray<T> operator/(const ndarray<T> &other) const;

  private:
    /**
     * @brief Internal constructor from shared tensor owner.
     *
     * @details Used by implementation helpers (for example `FromDLPack()` and
     * internal factory flows in `src/container/ndarray_typed.cpp`) to wrap an
     * already-managed tensor without additional allocation.
     */
    explicit ndarray(std::shared_ptr<DLManagedTensor> tensor);
};

} // namespace ndarray
