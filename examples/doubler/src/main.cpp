
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <type_traits>

#ifndef H_INTRINSIC_GENERATOR
#include "../../../src/intrinsic_generator.hh"
#define H_INTRINSIC_GENERATOR
#endif

#ifndef H_SIMD_CONTAINER
#include "../../../src/simd_container.hh"
#define H_SIMD_CONTAINER
#endif

template<typename T>
struct doubleit : UnrollerUnit<doubleit<T>, T> {

    template<typename VEC_T>
    inline void func(VEC_T x, VEC_T & y) {
        y = intr::add<T>()(x, x);
    }

    template<typename VEC_T>
    inline void maskfunc(VEC_T x, const unsigned int size, VEC_T & y) {
        auto mask = ~(intr::full_mask<T>()()  << size);
        y = intr::maskz_add<T>()(x, x, mask);
    }
};

using NUMBER_T = double;

int main() {
    const int N = 1<<16;
    NUMBER_T x[N];
    NUMBER_T y[N];
    for (int i = 0; i < N; ++i)
        x[i] = (NUMBER_T)i;

    simd_view<NUMBER_T> view(x, N);

    doubleit<NUMBER_T> d;

    view.process(d, y);

    auto index = rand() % N;
    std::cout << index << "*2 = " << y[index] << std::endl;
    std::cout << N-1 << "*2 = " << y[N-1] << std::endl;
}