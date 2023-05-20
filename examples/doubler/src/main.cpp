
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <type_traits>

#include "../../../src/simd_unroller.hh"
#include "../../../src/intrinsic_generator.hh"

template<typename T, int SZ>
struct doubleit {

    static auto x_init() {
        return vec_set1<T, SZ>()(0.0); 
    }

    static auto y_init() {
        return vec_set1<T, SZ>()(0.0); 
    }

    template<typename VEC_T>
    static void func(VEC_T x, VEC_T & y) {
        y = vec_add<T, SZ>()(x, x);
    }

    template<typename VEC_T>
    static void maskfunc(VEC_T x, const unsigned int size, VEC_T & y) {
        __mmask8 mask = vec_full_mask<T, SZ>()()  << size;
        y = vec_maskz_add<T, SZ>()(x, x, ~mask);
    }

    static auto load(T*from, int i) {
        return vec_loadu<T, SZ>()(from + i);
    }
    
    static auto maskload(T*from, int i, const unsigned int size) {
        auto mask = vec_full_mask<T, SZ>()() << size;
        return vec_maskz_loadu<T, SZ>()(from + i, ~mask);
    }

    template<typename VEC_T>
    static void store(VEC_T& x, int i, T*to) {
        vec_storeu<T, SZ>()(to + i, x);
    }

    template<typename VEC_T>
    static void maskstore(VEC_T x, const unsigned int size, int i, T*to) {
        __mmask8 mask = vec_full_mask<T, SZ>()() << size;
        vec_mask_storeu<T, SZ>()(to + i, x, ~mask);
    }

    template<typename VEC_T>
    static void reduce(VEC_T x, T*to) {
        
    }

    template<typename VEC_T>
    static void reduce(VEC_T x0, VEC_T x1, VEC_T x2, VEC_T& y) {
        
    }
};

using NUMBER_T = int;
const int SIZE = 512;

int main() {
    const int N = 1<<16;
    NUMBER_T x[N];
    NUMBER_T y[N];
    for (int i = 0; i < N; ++i)
        x[i] = (NUMBER_T)i;

    unroller<doubleit<NUMBER_T, SIZE>>(x, y, N);

    auto index = rand() % N;
    std::cout << index << "*2 = " << y[index] << std::endl;
    std::cout << N-1 << "*2 = " << y[N-1] << std::endl;
}