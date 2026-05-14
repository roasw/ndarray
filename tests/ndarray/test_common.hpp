#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace NdarrayTestCommon {

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
template <> constexpr double Epsilon<bool>() { return 0.0; }

template <typename T> double DiffMagnitude(T a, T b) {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return std::abs(a - b);
    } else {
        return std::fabs(static_cast<double>(a) - static_cast<double>(b));
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

} // namespace NdarrayTestCommon
