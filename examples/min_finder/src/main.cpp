
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <limits>

#ifndef H_INTRINSIC_GENERATOR
#include "../../../src/intrinsic_generator.hh"
#define H_INTRINSIC_GENERATOR
#endif

#ifndef H_SIMD_CONTAINER
#include "../../../src/simd_container.hh"
#define H_SIMD_CONTAINER
#endif

template<typename T>
struct min_finder  : UnrollerUnit<min_finder<T>, T>  {

    inline auto y_init() {
        return intr::set1<T>()(std::numeric_limits<T>::max());
    }

    template<typename VEC_T>
    inline void func(VEC_T x, VEC_T & y) {
        y = intr::min<T>()(x, y);
    }

    template<typename VEC_T>
    inline void maskfunc(VEC_T x, const unsigned int size, VEC_T & y) {
        auto mask = ~(intr::full_mask<T>()()  << size);
        y = intr::mask_min<T>()(y, x, y, mask);
    }

    template<typename VEC_T>
    inline auto store(T*to, VEC_T x) 
    {

    }

    template<typename VEC_T>
    inline auto maskstore(T*to, VEC_T x, unsigned int places) 
    {

    }

    template<typename VEC_T>
    inline void reduce(VEC_T x, T* to) {
        (*to) = intr::reduce_min<T>()(x);
    }

    template<typename VEC_T>
    inline void reduce(VEC_T x0, VEC_T x1, VEC_T x2, VEC_T& y) {
        x0 = intr::min<T>()(x0, x1);
        x2 = intr::min<T>()(x2, y);
        y = intr::min<T>()(x0, x2);
    }
};

using NUMBER_T = float;

int main() {
    const int N = 15615;
    auto n = rand() % N;
    NUMBER_T x[n];
    NUMBER_T y = 0.0;
    for (int i = 0; i < n; ++i)
        x[i] = (NUMBER_T)(rand() % N) + 2;

    simd_view<NUMBER_T> view(x, n);
    min_finder<NUMBER_T> m;

    view.process(m, &y);

    std::cout << y << std::endl;
}