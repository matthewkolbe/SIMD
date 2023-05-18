
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <experimental/simd>

#include"../../../src/simd_unroller.hh"

namespace stdx = std::experimental;

struct stl_halve {
    static auto x_init() {
        stdx::native_simd<double> r{0.0};
        return r;
    }

    static auto y_init() {
        stdx::native_simd<double> r{0.0};
        return r;
    }

    static void func(stdx::native_simd<double> x, stdx::native_simd<double> & y) {
        y = x / 2.0;
    }

    static void maskfunc(stdx::native_simd<double> x, const unsigned int size, stdx::native_simd<double> & y) {
        // i don't know how to mask with this library
    }

    static auto load(double*from) {
        stdx::native_simd<double> r(from, stdx::element_aligned);
        return r;
    }

    static auto maskload(double*from, const unsigned int size) {
        // i don't know how to mask with this library
        stdx::native_simd<double> r(from, stdx::element_aligned);
        return r;
    }

    static void store(stdx::native_simd<double> x, double*to) {
        x.copy_to(to, stdx::element_aligned);
    }

    static void maskstore(stdx::native_simd<double> x, const unsigned int size, double*to) {
        // i don't know how to mask with this library
    }

    constexpr static bool reduce_is_valid() {
        return false;
    }

    static void reduce(stdx::native_simd<double> x, double*to) {
    }

    static void reduce(stdx::native_simd<double> x0, stdx::native_simd<double> x1, stdx::native_simd<double> x2, stdx::native_simd<double>& y) {
    }
};



int main() {
    const int N = 4096;
    double x[N];
    double y[N];
    for (int i = 0; i < N; ++i)
        x[i] = i;

    unroller<stl_halve>(x, y, N);

    auto idx = rand() % N;
    std::cout << x[idx] << "/2 = " << y[idx] << std::endl;
}