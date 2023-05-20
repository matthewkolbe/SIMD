
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <experimental/simd>

#include"../../../src/simd_unroller.hh"

namespace stdx = std::experimental;

template<typename T>
struct stl_halve {
    constexpr static auto x_init() {
        stdx::native_simd<T> r{(T)0};
        return r;
    }

    constexpr static auto y_init() {
        stdx::native_simd<T> r{(T)0};
        return r;
    }

    static void func(stdx::native_simd<T> x, stdx::native_simd<T> & y) {
        y = x / (T)2;
    }

    static void maskfunc(stdx::native_simd<T> x, const unsigned int size, stdx::native_simd<T> & y) {
        // i don't know how to mask with this library
    }

    static auto load(T*from) {
        stdx::native_simd<T> r(from, stdx::element_aligned);
        return r;
    }

    static auto maskload(T*from, const unsigned int size) {
        // i don't know how to mask with this library
        stdx::native_simd<T> r(from, stdx::element_aligned);
        return r;
    }

    static void store(stdx::native_simd<T> x, T*to) {
        x.copy_to(to, stdx::element_aligned);
    }

    static void maskstore(stdx::native_simd<T> x, const unsigned int size, T*to) {
        // i don't know how to mask with this library
    }

    static void reduce(stdx::native_simd<T> x, T*to) {
    }

    static void reduce(stdx::native_simd<T> x0, stdx::native_simd<T> x1, stdx::native_simd<T> x2, stdx::native_simd<T>& y) {
    }
};

using NUMBER_T = float;

int main() {
    const int N = 1<<16;
    NUMBER_T x[N];
    NUMBER_T y[N];
    for (int i = 0; i < N; ++i)
        x[i] = i;

    unroller<stl_halve<NUMBER_T>>(x, y, N);

    auto idx = rand() % N;
    std::cout << x[idx] << "/2 = " << y[idx] << std::endl;
}