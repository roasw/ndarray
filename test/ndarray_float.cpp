#include "ndarray.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/dlpack.h>
#include <armadillo>

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) static void name()

#define RUN(name)                                                              \
    do {                                                                       \
        try {                                                                  \
            name();                                                            \
            std::cout << "  PASS  " #name "\n";                                \
            ++g_passed;                                                        \
        } catch (const std::exception &e) {                                    \
            std::cout << "  FAIL  " #name " — " << e.what() << "\n";           \
            ++g_failed;                                                        \
        } catch (...) {                                                        \
            std::cout << "  FAIL  " #name " — unknown exception\n";            \
            ++g_failed;                                                        \
        }                                                                      \
    } while (0)

static void require(bool cond, const std::string &msg) {
    if (!cond)
        throw std::runtime_error(msg);
}

static void require_near(float a, float b, const std::string &msg,
                         float eps = 1e-5f) {
    if (std::fabs(a - b) > eps)
        throw std::runtime_error(msg + " (" + std::to_string(a) +
                                 " != " + std::to_string(b) + ")");
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(test_default_construction) {
    ndarray::ndarray<float> a;
    require(a.GetNdim() == 0, "default ndim != 0");
    require(a.GetData() == nullptr, "default data != nullptr");
    require(a.GetShape().empty(), "default shape not empty");
    require(a.GetStrides().empty(), "default strides not empty");
}

TEST(test_1d_construction) {
    ndarray::ndarray<float> a({4});
    require(a.GetNdim() == 1, "1D ndim");
    require(a.GetShape() == std::vector<int64_t>{4}, "1D shape");
    require(a.GetStrides() == std::vector<int64_t>{1}, "1D strides");
    require(a.GetData() != nullptr, "1D data null");
}

TEST(test_2d_construction) {
    ndarray::ndarray<float> a({3, 5});
    require(a.GetNdim() == 2, "2D ndim");
    require(a.GetShape() == (std::vector<int64_t>{3, 5}), "2D shape");
    // column-major: strides = [1, rows]
    require(a.GetStrides() == (std::vector<int64_t>{1, 3}),
            "2D col-major strides");
    require(a.GetData() != nullptr, "2D data null");
}

TEST(test_3d_construction) {
    ndarray::ndarray<float> a({2, 3, 4});
    require(a.GetNdim() == 3, "3D ndim");
    require(a.GetShape() == (std::vector<int64_t>{2, 3, 4}), "3D shape");
    // column-major: strides = [1, 2, 6]
    require(a.GetStrides() == (std::vector<int64_t>{1, 2, 6}),
            "3D col-major strides");
}

TEST(test_device_cpu) {
    ndarray::ndarray<float> a({2, 2});
    require(a.GetDevice() == c10::DeviceType::CPU, "device should be CPU");
}

// ---------------------------------------------------------------------------
// Value semantics — copy is zero-copy (shared buffer)
// ---------------------------------------------------------------------------

TEST(test_copy_shares_buffer) {
    ndarray::ndarray<float> a({2, 2});
    a.At(int64_t(0), int64_t(0)) = 1.0f;

    ndarray::ndarray<float> b = a; // copy — shared ownership
    require(b.GetData() == a.GetData(), "copy should share data pointer");

    b.At(int64_t(0), int64_t(0)) = 99.0f;
    require_near(a.At(int64_t(0), int64_t(0)), 99.0f,
                 "mutation through copy should be visible");
}

TEST(test_copy_assign_shares_buffer) {
    ndarray::ndarray<float> a({3, 3});
    a.At(int64_t(1), int64_t(1)) = 7.0f;

    ndarray::ndarray<float> b;
    b = a;
    require(b.GetData() == a.GetData(),
            "copy-assign should share data pointer");
    require_near(b.At(int64_t(1), int64_t(1)), 7.0f, "copy-assign value");
}

TEST(test_move_transfers_ownership) {
    ndarray::ndarray<float> a({2, 3});
    float *ptr = a.GetData();

    ndarray::ndarray<float> b = std::move(a);
    require(b.GetData() == ptr, "move should transfer data pointer");
    require(a.GetData() == nullptr, "moved-from should have null data");
}

TEST(test_move_assign_transfers_ownership) {
    ndarray::ndarray<float> a({4, 2});
    float *ptr = a.GetData();

    ndarray::ndarray<float> b({1, 1});
    b = std::move(a);
    require(b.GetData() == ptr, "move-assign should transfer data pointer");
    require(a.GetData() == nullptr, "moved-from should have null data");
}

// ---------------------------------------------------------------------------
// Clone — explicit deep copy
// ---------------------------------------------------------------------------

TEST(test_clone_independent_buffer) {
    ndarray::ndarray<float> a({3, 3});
    a.At(int64_t(0), int64_t(0)) = 5.0f;

    ndarray::ndarray<float> b = a.Clone();
    require(b.GetData() != a.GetData(),
            "clone should have distinct data pointer");
    require_near(b.At(int64_t(0), int64_t(0)), 5.0f,
                 "clone should copy values");

    b.At(int64_t(0), int64_t(0)) = 99.0f;
    require_near(a.At(int64_t(0), int64_t(0)), 5.0f,
                 "mutation of clone should not affect original");
}

TEST(test_clone_preserves_shape) {
    ndarray::ndarray<float> a({4, 7});
    ndarray::ndarray<float> b = a.Clone();
    require(b.GetShape() == a.GetShape(), "clone shape");
    require(b.GetStrides() == a.GetStrides(), "clone strides");
    require(b.GetNdim() == a.GetNdim(), "clone ndim");
}

TEST(test_clone_of_default_is_default) {
    ndarray::ndarray<float> a;
    ndarray::ndarray<float> b = a.Clone();
    require(b.GetData() == nullptr, "clone of empty should be empty");
    require(b.GetNdim() == 0, "clone of empty ndim");
}

// ---------------------------------------------------------------------------
// At() — element access
// ---------------------------------------------------------------------------

TEST(test_at_1d_roundtrip) {
    ndarray::ndarray<float> a({5});
    for (int64_t i = 0; i < 5; ++i) {
        a.At(i) = static_cast<float>(i * 10);
    }
    for (int64_t i = 0; i < 5; ++i) {
        require_near(a.At(i), static_cast<float>(i * 10), "1D At roundtrip");
    }
}

TEST(test_at_2d_roundtrip) {
    ndarray::ndarray<float> a({3, 4});
    for (int64_t r = 0; r < 3; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            a.At(r, c) = static_cast<float>(r * 10 + c);
        }
    }
    for (int64_t r = 0; r < 3; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            require_near(a.At(r, c), static_cast<float>(r * 10 + c),
                         "2D At roundtrip");
        }
    }
}

TEST(test_at_2d_column_major_layout) {
    // Verify column-major memory layout: a(r,c) lives at data[r + c*rows]
    ndarray::ndarray<float> a({3, 4});
    float *data = a.GetData();
    for (int64_t i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }
    // data[0]=a(0,0), data[1]=a(1,0), data[2]=a(2,0), data[3]=a(0,1) ...
    require_near(a.At(int64_t(0), int64_t(0)), 0.0f, "col-major (0,0)");
    require_near(a.At(int64_t(1), int64_t(0)), 1.0f, "col-major (1,0)");
    require_near(a.At(int64_t(2), int64_t(0)), 2.0f, "col-major (2,0)");
    require_near(a.At(int64_t(0), int64_t(1)), 3.0f, "col-major (0,1)");
    require_near(a.At(int64_t(1), int64_t(1)), 4.0f, "col-major (1,1)");
    require_near(a.At(int64_t(0), int64_t(3)), 9.0f, "col-major (0,3)");
}

TEST(test_at_const) {
    ndarray::ndarray<float> a({2, 2});
    a.At(int64_t(0), int64_t(1)) = 3.14f;
    const ndarray::ndarray<float> &ca = a;
    require_near(ca.At(int64_t(0), int64_t(1)), 3.14f, "const At");
}

TEST(test_at_empty_throws) {
    ndarray::ndarray<float> a;
    bool threw = false;
    try {
        a.At(int64_t(0));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "At on empty should throw");
}

TEST(test_at_wrong_ndim_throws) {
    ndarray::ndarray<float> a({3, 3});
    bool threw = false;
    try {
        a.At(int64_t(0)); // 1D access on 2D array
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "At with wrong ndim should throw");
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

TEST(test_transpose_shape) {
    ndarray::ndarray<float> a({3, 5});
    ndarray::ndarray<float> t = a.Transpose();
    require(t.GetShape() == (std::vector<int64_t>{5, 3}), "transpose shape");
}

TEST(test_transpose_values) {
    ndarray::ndarray<float> a({2, 3});
    // Fill: a(r,c) = r*3 + c
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c = 0; c < 3; ++c)
            a.At(r, c) = static_cast<float>(r * 3 + c);

    ndarray::ndarray<float> t = a.Transpose();
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c = 0; c < 3; ++c)
            require_near(t.At(c, r), a.At(r, c), "transpose value");
}

TEST(test_transpose_independent_buffer) {
    ndarray::ndarray<float> a({3, 4});
    ndarray::ndarray<float> t = a.Transpose();
    require(t.GetData() != a.GetData(),
            "transpose should allocate a new buffer");
}

TEST(test_transpose_non_2d_throws) {
    ndarray::ndarray<float> a({4});
    bool threw = false;
    try {
        a.Transpose();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "Transpose on 1D should throw");
}

// ---------------------------------------------------------------------------
// AsArmadillo — lifetime-extending view
// ---------------------------------------------------------------------------

TEST(test_arma_view_shares_data) {
    ndarray::ndarray<float> a({3, 4});
    a.At(int64_t(1), int64_t(2)) = 42.0f;

    auto view = a.AsArmadillo();
    require_near(view.mat(1, 2), 42.0f, "arma view reads ndarray data");
}

TEST(test_arma_view_mutation_visible_in_ndarray) {
    ndarray::ndarray<float> a({3, 4});
    auto view = a.AsArmadillo();
    view.mat(0, 0) = 77.0f;
    require_near(a.At(int64_t(0), int64_t(0)), 77.0f,
                 "arma mutation visible in ndarray");
}

TEST(test_arma_view_extends_lifetime) {
    ndarray::ndarray<float> *a = new ndarray::ndarray<float>({3, 3});
    a->At(int64_t(2), int64_t(2)) = 55.0f;

    auto view = a->AsArmadillo();
    delete a; // ndarray destroyed — view should keep buffer alive

    // If lifetime management is broken this is a use-after-free.
    // Under sanitizers (ASan) this would trap immediately.
    require_near(view.mat(2, 2), 55.0f,
                 "arma view still valid after ndarray destroyed");
}

TEST(test_arma_view_implicit_conversion) {
    ndarray::ndarray<float> a({2, 2});
    a.At(int64_t(0), int64_t(0)) = 1.0f;
    a.At(int64_t(1), int64_t(0)) = 2.0f;
    a.At(int64_t(0), int64_t(1)) = 3.0f;
    a.At(int64_t(1), int64_t(1)) = 4.0f;

    auto view = a.AsArmadillo();
    arma::Mat<float> &m = view; // implicit conversion
    require_near(m(0, 0), 1.0f, "implicit conversion (0,0)");
    require_near(m(1, 1), 4.0f, "implicit conversion (1,1)");
}

TEST(test_arma_view_non_2d_throws) {
    ndarray::ndarray<float> a({6});
    bool threw = false;
    try {
        a.AsArmadillo();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "AsArmadillo on 1D should throw");
}

TEST(test_arma_view_const) {
    ndarray::ndarray<float> a({2, 3});
    a.At(int64_t(0), int64_t(2)) = 9.0f;
    const ndarray::ndarray<float> &ca = a;
    auto view = ca.AsArmadillo();
    require_near(view.mat(0, 2), 9.0f, "const AsArmadillo");
}

// ---------------------------------------------------------------------------
// ToDLPack — zero-copy export
// ---------------------------------------------------------------------------

TEST(test_dlpack_null_for_empty) {
    ndarray::ndarray<float> a;
    require(a.ToDLPack() == nullptr, "ToDLPack on empty should return null");
}

TEST(test_dlpack_zero_copy) {
    ndarray::ndarray<float> a({3, 4});
    DLManagedTensor *mt = a.ToDLPack();
    require(mt != nullptr, "ToDLPack returned null");
    require(mt->dl_tensor.data == a.GetData(),
            "DLPack data pointer must equal ndarray data pointer");
    mt->deleter(mt);
}

TEST(test_dlpack_metadata) {
    ndarray::ndarray<float> a({3, 4});
    DLManagedTensor *mt = a.ToDLPack();

    require(mt->dl_tensor.ndim == 2, "DLPack ndim");
    require(mt->dl_tensor.shape[0] == 3, "DLPack shape[0]");
    require(mt->dl_tensor.shape[1] == 4, "DLPack shape[1]");
    require(mt->dl_tensor.strides[0] == 1, "DLPack strides[0]");
    require(mt->dl_tensor.strides[1] == 3, "DLPack strides[1]");
    require(mt->dl_tensor.dtype.code == kDLFloat, "DLPack dtype code");
    require(mt->dl_tensor.dtype.bits == 32, "DLPack dtype bits");
    require(mt->dl_tensor.device.device_type == kDLCPU, "DLPack device");

    mt->deleter(mt);
}

TEST(test_dlpack_extends_lifetime) {
    DLManagedTensor *mt = nullptr;
    {
        ndarray::ndarray<float> a({3, 3});
        a.At(int64_t(0), int64_t(0)) = 11.0f;
        mt = a.ToDLPack();
    } // ndarray destroyed here

    // Buffer must still be alive because DLManagedTensor holds a shared_ptr
    // bump.
    require_near(static_cast<float *>(mt->dl_tensor.data)[0], 11.0f,
                 "DLPack buffer alive after ndarray destroyed");
    mt->deleter(mt);
}

TEST(test_dlpack_multiple_exports_share_buffer) {
    ndarray::ndarray<float> a({2, 2});
    a.At(int64_t(0), int64_t(0)) = 3.0f;

    DLManagedTensor *mt1 = a.ToDLPack();
    DLManagedTensor *mt2 = a.ToDLPack();

    require(mt1->dl_tensor.data == mt2->dl_tensor.data,
            "multiple DLPack exports share same buffer");

    mt1->deleter(mt1);
    // Buffer still alive — mt2 holds a ref.
    require_near(static_cast<float *>(mt2->dl_tensor.data)[0], 3.0f,
                 "buffer alive after first DLPack released");
    mt2->deleter(mt2);
}

// ---------------------------------------------------------------------------
// Arithmetic — Add, Subtract, Multiply, Divide
// ---------------------------------------------------------------------------

TEST(test_add_2d) {
    ndarray::ndarray<float> a({2, 2}), b({2, 2});
    a.At(int64_t(0), int64_t(0)) = 1.0f;
    a.At(int64_t(1), int64_t(0)) = 2.0f;
    a.At(int64_t(0), int64_t(1)) = 3.0f;
    a.At(int64_t(1), int64_t(1)) = 4.0f;
    b.At(int64_t(0), int64_t(0)) = 10.0f;
    b.At(int64_t(1), int64_t(0)) = 20.0f;
    b.At(int64_t(0), int64_t(1)) = 30.0f;
    b.At(int64_t(1), int64_t(1)) = 40.0f;

    ndarray::ndarray<float> c = a + b;
    require_near(c.At(int64_t(0), int64_t(0)), 11.0f, "add (0,0)");
    require_near(c.At(int64_t(1), int64_t(0)), 22.0f, "add (1,0)");
    require_near(c.At(int64_t(0), int64_t(1)), 33.0f, "add (0,1)");
    require_near(c.At(int64_t(1), int64_t(1)), 44.0f, "add (1,1)");
}

TEST(test_subtract_2d) {
    ndarray::ndarray<float> a({2, 2}), b({2, 2});
    a.At(int64_t(0), int64_t(0)) = 10.0f;
    a.At(int64_t(1), int64_t(0)) = 20.0f;
    b.At(int64_t(0), int64_t(0)) = 3.0f;
    b.At(int64_t(1), int64_t(0)) = 4.0f;

    ndarray::ndarray<float> c = a - b;
    require_near(c.At(int64_t(0), int64_t(0)), 7.0f, "subtract (0,0)");
    require_near(c.At(int64_t(1), int64_t(0)), 16.0f, "subtract (1,0)");
}

TEST(test_multiply_elementwise_2d) {
    ndarray::ndarray<float> a({2, 2}), b({2, 2});
    a.At(int64_t(0), int64_t(0)) = 2.0f;
    a.At(int64_t(1), int64_t(1)) = 3.0f;
    b.At(int64_t(0), int64_t(0)) = 4.0f;
    b.At(int64_t(1), int64_t(1)) = 5.0f;

    ndarray::ndarray<float> c = a * b;
    require_near(c.At(int64_t(0), int64_t(0)), 8.0f, "multiply (0,0)");
    require_near(c.At(int64_t(1), int64_t(1)), 15.0f, "multiply (1,1)");
}

TEST(test_divide_elementwise_2d) {
    ndarray::ndarray<float> a({2, 2}), b({2, 2});
    a.At(int64_t(0), int64_t(0)) = 9.0f;
    a.At(int64_t(0), int64_t(1)) = 6.0f;
    b.At(int64_t(0), int64_t(0)) = 3.0f;
    b.At(int64_t(0), int64_t(1)) = 2.0f;

    ndarray::ndarray<float> c = a / b;
    require_near(c.At(int64_t(0), int64_t(0)), 3.0f, "divide (0,0)");
    require_near(c.At(int64_t(0), int64_t(1)), 3.0f, "divide (0,1)");
}

TEST(test_arithmetic_result_is_independent) {
    ndarray::ndarray<float> a({2, 2}), b({2, 2});
    a.At(int64_t(0), int64_t(0)) = 1.0f;
    b.At(int64_t(0), int64_t(0)) = 1.0f;

    ndarray::ndarray<float> c = a + b;
    require(c.GetData() != a.GetData(), "result must not alias a");
    require(c.GetData() != b.GetData(), "result must not alias b");
}

TEST(test_arithmetic_shape_mismatch_throws) {
    ndarray::ndarray<float> a({2, 3}), b({3, 2});
    bool threw = false;
    try {
        auto c = a + b;
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "shape mismatch should throw");
}

TEST(test_arithmetic_on_empty_throws) {
    ndarray::ndarray<float> a({2, 2}), b;
    bool threw = false;
    try {
        auto c = a + b;
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "arithmetic with empty should throw");
}

TEST(test_arithmetic_non_2d_throws) {
    ndarray::ndarray<float> a({4}), b({4});
    bool threw = false;
    try {
        auto c = a + b;
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "arithmetic on 1D should throw");
}

// ---------------------------------------------------------------------------
// Shared ownership — buffer stays alive across multiple handles
// ---------------------------------------------------------------------------

TEST(test_shared_ownership_refcount) {
    ndarray::ndarray<float> a({4, 4});
    a.At(int64_t(0), int64_t(0)) = 123.0f;

    {
        ndarray::ndarray<float> b = a;
        ndarray::ndarray<float> c = b;
        // All three share the same buffer.
        require(a.GetData() == b.GetData(), "shared owner a==b");
        require(b.GetData() == c.GetData(), "shared owner b==c");
        c.At(int64_t(0), int64_t(0)) = 456.0f;
        require_near(a.At(int64_t(0), int64_t(0)), 456.0f,
                     "mutation visible across all handles");
    }
    // b and c gone; a still valid
    require_near(a.At(int64_t(0), int64_t(0)), 456.0f,
                 "buffer alive after copies destroyed");
}

TEST(test_self_assign) {
    ndarray::ndarray<float> a({2, 2});
    a.At(int64_t(0), int64_t(0)) = 7.0f;
    a = a; // must not crash or corrupt
    require_near(a.At(int64_t(0), int64_t(0)), 7.0f,
                 "self-assign preserves value");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== ndarray<float> tests ===\n";

    // Construction
    RUN(test_default_construction);
    RUN(test_1d_construction);
    RUN(test_2d_construction);
    RUN(test_3d_construction);
    RUN(test_device_cpu);

    // Value semantics
    RUN(test_copy_shares_buffer);
    RUN(test_copy_assign_shares_buffer);
    RUN(test_move_transfers_ownership);
    RUN(test_move_assign_transfers_ownership);

    // Clone
    RUN(test_clone_independent_buffer);
    RUN(test_clone_preserves_shape);
    RUN(test_clone_of_default_is_default);

    // At()
    RUN(test_at_1d_roundtrip);
    RUN(test_at_2d_roundtrip);
    RUN(test_at_2d_column_major_layout);
    RUN(test_at_const);
    RUN(test_at_empty_throws);
    RUN(test_at_wrong_ndim_throws);

    // Transpose
    RUN(test_transpose_shape);
    RUN(test_transpose_values);
    RUN(test_transpose_independent_buffer);
    RUN(test_transpose_non_2d_throws);

    // AsArmadillo
    RUN(test_arma_view_shares_data);
    RUN(test_arma_view_mutation_visible_in_ndarray);
    RUN(test_arma_view_extends_lifetime);
    RUN(test_arma_view_implicit_conversion);
    RUN(test_arma_view_non_2d_throws);
    RUN(test_arma_view_const);

    // ToDLPack
    RUN(test_dlpack_null_for_empty);
    RUN(test_dlpack_zero_copy);
    RUN(test_dlpack_metadata);
    RUN(test_dlpack_extends_lifetime);
    RUN(test_dlpack_multiple_exports_share_buffer);

    // Arithmetic
    RUN(test_add_2d);
    RUN(test_subtract_2d);
    RUN(test_multiply_elementwise_2d);
    RUN(test_divide_elementwise_2d);
    RUN(test_arithmetic_result_is_independent);
    RUN(test_arithmetic_shape_mismatch_throws);
    RUN(test_arithmetic_on_empty_throws);
    RUN(test_arithmetic_non_2d_throws);

    // Shared ownership
    RUN(test_shared_ownership_refcount);
    RUN(test_self_assign);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed > 0 ? 1 : 0;
}
