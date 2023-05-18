
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <limits>

#include"../../../src/simd_unroller.hh"

struct min_finder {

    static auto x_init() {
        return _mm512_set1_pd(0.0); 
    }

    static auto y_init() {
        return _mm512_set1_pd(std::numeric_limits<double>::max());
    }

    static void func(__m512d x, __m512d & y) {
        y = _mm512_min_pd(x, y);
    }

    static void maskfunc(__m512d x, const unsigned int size, __m512d & y) {
        unsigned char mask = 0xFF << size;
        y = _mm512_mask_min_pd(y, ~mask, x, y);
    }

    static __m512d load(double*from) {
        return _mm512_loadu_pd(from);
    }

    static __m512d maskload(double*from, const unsigned int size) {
        unsigned char mask = 0xFF << size;
        return _mm512_maskz_loadu_pd(~mask, from);
    }

    static void store(__m512d x, double* to) {
    }

    static void maskstore(__m512d x, const unsigned int size, double* to) {
    }

    constexpr static bool reduce_is_valid() {
        return true;
    }

    static void reduce(__m512d x, double* to) {
        (*to) = _mm512_reduce_min_pd(x);
    }

    static void reduce(__m512d x0, __m512d x1, __m512d x2, __m512d& y) {
        x0 = _mm512_min_pd(x0, x1);
        x2 = _mm512_min_pd(x2, y);
        y = _mm512_min_pd(x0, x2);
    }
};



int main() {
    const int N = 35128;
    auto n = rand() % N;
    double x[n];
    double y = 0.0;
    for (int i = 0; i < n; ++i)
        x[i] = (rand() % N);

    unroller<min_finder>(x, &y, n);

    std::cout << y << std::endl;
}