#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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
template <typename T> double DiffMagnitude(T a, T b) {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return std::abs(a - b);
    } else {
        return std::fabs(static_cast<double>(a) - static_cast<double>(b));
    }
}

template <typename T>
void RequireNear(T a, T b, const std::string &msg, double eps = Epsilon<T>()) {
    if (DiffMagnitude(a, b) > eps) {
        throw std::runtime_error(msg);
    }
}

inline void Require(bool cond, const std::string &msg) {
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

/**
 * @brief Populate deterministic 2x2 inputs for arithmetic verification.
 *
 * @details Fixed non-zero values make expected add/sub/mul/div outcomes
 * unambiguous and keep division safe from zero-denominator edge cases.
 */
template <typename T>
void FillInputs(ndarray::ndarray<T> &a, ndarray::ndarray<T> &b) {
    a.At(int64_t(0), int64_t(0)) = static_cast<T>(8);
    a.At(int64_t(1), int64_t(0)) = static_cast<T>(10);
    a.At(int64_t(0), int64_t(1)) = static_cast<T>(9);
    a.At(int64_t(1), int64_t(1)) = static_cast<T>(11);

    b.At(int64_t(0), int64_t(0)) = static_cast<T>(2);
    b.At(int64_t(1), int64_t(0)) = static_cast<T>(4);
    b.At(int64_t(0), int64_t(1)) = static_cast<T>(3);
    b.At(int64_t(1), int64_t(1)) = static_cast<T>(5);
}

/**
 * @brief Verify element-wise arithmetic values against scalar reference math.
 *
 * @details The test computes `+`, `-`, `*`, and `/` on ndarrays, then compares
 * each output element with scalar expressions on input elements. This works as
 * a direct semantic oracle for element-wise behavior across supported dtypes.
 */
template <typename T> void TestArithmeticValues() {
    ndarray::ndarray<T> a({2, 2});
    ndarray::ndarray<T> b({2, 2});
    FillInputs(a, b);

    const ndarray::ndarray<T> sum = a + b;
    const ndarray::ndarray<T> diff = a - b;
    const ndarray::ndarray<T> prod = a * b;
    const ndarray::ndarray<T> quot = a / b;

    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t r = 0; r < 2; ++r) {
            const T av = a.At(r, c);
            const T bv = b.At(r, c);
            RequireNear(sum.At(r, c), static_cast<T>(av + bv), "add mismatch");
            RequireNear(diff.At(r, c), static_cast<T>(av - bv), "sub mismatch");
            RequireNear(prod.At(r, c), static_cast<T>(av * bv), "mul mismatch");
            RequireNear(quot.At(r, c), static_cast<T>(av / bv), "div mismatch");
        }
    }
}

/**
 * @brief Validate bool arithmetic policy (OR/AND; no subtract/divide).
 *
 * @details For bool, `+` maps to logical OR and `*` maps to logical AND by
 * project contract. Subtract and divide must throw. This test asserts all
 * three policy points explicitly.
 */
template <> void TestArithmeticValues<bool>() {
    ndarray::ndarray<bool> a({2, 2});
    ndarray::ndarray<bool> b({2, 2});

    a.At(int64_t(0), int64_t(0)) = true;
    a.At(int64_t(1), int64_t(0)) = false;
    a.At(int64_t(0), int64_t(1)) = false;
    a.At(int64_t(1), int64_t(1)) = true;

    b.At(int64_t(0), int64_t(0)) = false;
    b.At(int64_t(1), int64_t(0)) = true;
    b.At(int64_t(0), int64_t(1)) = true;
    b.At(int64_t(1), int64_t(1)) = true;

    const ndarray::ndarray<bool> sum = a + b;
    const ndarray::ndarray<bool> prod = a * b;

    Require(sum.At(int64_t(0), int64_t(0)) == true, "bool add (0,0)");
    Require(sum.At(int64_t(1), int64_t(0)) == true, "bool add (1,0)");
    Require(sum.At(int64_t(0), int64_t(1)) == true, "bool add (0,1)");
    Require(sum.At(int64_t(1), int64_t(1)) == true, "bool add (1,1)");

    Require(prod.At(int64_t(0), int64_t(0)) == false, "bool mul (0,0)");
    Require(prod.At(int64_t(1), int64_t(0)) == false, "bool mul (1,0)");
    Require(prod.At(int64_t(0), int64_t(1)) == false, "bool mul (0,1)");
    Require(prod.At(int64_t(1), int64_t(1)) == true, "bool mul (1,1)");

    bool subtractThrew = false;
    try {
        (void)(a - b);
    } catch (const std::runtime_error &) {
        subtractThrew = true;
    }
    Require(subtractThrew, "bool subtract should throw");

    bool divideThrew = false;
    try {
        (void)(a / b);
    } catch (const std::runtime_error &) {
        divideThrew = true;
    }
    Require(divideThrew, "bool divide should throw");
}

/**
 * @brief Ensure arithmetic outputs do not alias input storage.
 *
 * @details The test checks output pointer inequality vs both operands and then
 * mutates output to verify lhs remains unchanged. This proves result allocation
 * is independent even when arithmetic is implemented through torch ops.
 */
template <typename T> void TestResultStorageIndependent() {
    ndarray::ndarray<T> a({2, 2});
    ndarray::ndarray<T> b({2, 2});
    FillInputs(a, b);

    ndarray::ndarray<T> out = a + b;
    Require(out.GetData() != a.GetData(), "result should not alias lhs");
    Require(out.GetData() != b.GetData(), "result should not alias rhs");

    const T lhsBefore = a.At(int64_t(0), int64_t(0));
    out.At(int64_t(0), int64_t(0)) = static_cast<T>(123);
    RequireNear(a.At(int64_t(0), int64_t(0)), lhsBefore,
                "mutating result should not affect lhs");
}

/**
 * @brief Bool specialization of non-aliasing result storage verification.
 *
 * @details Mirrors non-bool independence checks for bool OR path and confirms
 * output mutation does not write through into the source operand.
 */
template <> void TestResultStorageIndependent<bool>() {
    ndarray::ndarray<bool> a({2, 2});
    ndarray::ndarray<bool> b({2, 2});
    a.At(int64_t(0), int64_t(0)) = true;
    b.At(int64_t(0), int64_t(0)) = false;

    ndarray::ndarray<bool> out = a + b;
    Require(out.GetData() != a.GetData(), "result should not alias lhs");
    Require(out.GetData() != b.GetData(), "result should not alias rhs");

    out.At(int64_t(0), int64_t(0)) = false;
    Require(a.At(int64_t(0), int64_t(0)) == true,
            "mutating bool result should not affect lhs");
}

/**
 * @brief Verify binary arithmetic rejects mismatched operand shapes.
 *
 * @details Constructing `{2,2}` and `{2,3}` operands and expecting throw
 * confirms shape validation gates arithmetic before dispatching compute
 * kernels.
 */
template <typename T> void TestShapeMismatchRejected() {
    ndarray::ndarray<T> a({2, 2});
    ndarray::ndarray<T> b({2, 3});

    bool threw = false;
    try {
        (void)(a + b);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    Require(threw, "shape mismatch should throw");
}

template <typename T> void RunTypedSuite(Counters &counters) {
    const std::string prefix =
        std::string("[") + std::string(TypeName<T>::value) + "] ";
    RunCase(prefix + "arithmetic_values", counters,
            [] { TestArithmeticValues<T>(); });
    RunCase(prefix + "result_storage_independent", counters,
            [] { TestResultStorageIndependent<T>(); });
    RunCase(prefix + "shape_mismatch_rejected", counters,
            [] { TestShapeMismatchRejected<T>(); });
}

} // namespace

int main() {
    std::cout << "=== ndarray arithmetic tests ===\n";

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
