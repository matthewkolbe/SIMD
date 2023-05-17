
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>

#include"../../../src/simd_unroller.hh"


struct doubleit {
    static void func(__m512d x, __m512d & y) {
        y = _mm512_add_pd(x, x);
    }

    static void maskfunc(__m512d x, const unsigned int size, __m512d & y) {
        unsigned char mask = 0xFF << size;
        y = _mm512_maskz_add_pd(~mask, x, x);
    }

    static __m512d load(double*from) {
        return _mm512_loadu_pd(from);
    }

    static __m512d maskload(double*from, const unsigned int size) {
        unsigned char mask = 0xFF << size;
        return _mm512_maskz_loadu_pd(~mask, from);
    }

    static void store(__m512d x, double*to) {
        _mm512_storeu_pd(to, x);
    }

    static void maskstore(__m512d x, const unsigned int size, double*to) {
        unsigned char mask = 0xFF << size;
        return _mm512_mask_storeu_pd(to, ~mask, x);
    }

    constexpr static bool reduce_is_valid() {
        return false;
    }

    static void reduce(__m512d x, double*to) {
        
    }

    static void reduce(__m512d x0, __m512d x1, __m512d x2, __m512d& y) {
        
    }
};



int main() {
    const int N = 35128;
    auto n = rand() % N;
    double x[n];
    double y[n];
    for (int i = 0; i < n; ++i)
        x[i] = (double)i;

    unroller<double, double, __m512d, __m512d, doubleit>(x, y, n);

    auto index = rand() % n;
    std::cout << index << "*2 = " << y[index] << std::endl;
    std::cout << n-1 << "*2 = " << y[n-1] << std::endl;
}