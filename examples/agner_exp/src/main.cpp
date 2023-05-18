
#include <immintrin.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <vectorclass.h>
#include "vectormath_exp.h"

#include"../../../src/simd_unroller.hh"


struct agner_exp {
    static auto x_init() {
        Vec8d r{0.0};
        return r;
    }

    static auto y_init() {
        Vec8d r{0.0};
        return r;
    }

    static void func(Vec8d x, Vec8d & y) {
        y = exp(x);
    }

    static void maskfunc(Vec8d x, const unsigned int size, Vec8d & y) {
        y = exp(x);
    }

    static auto load(double*from) {
        Vec8d r;
        return r.load(from);
    }

    static auto maskload(double*from, const unsigned int size) {
        Vec8d r;
        return r.load_partial(size, from);
    }

    static void store(Vec8d x, double*to) {
        x.store(to);
    }

    static void maskstore(Vec8d x, const unsigned int size, double*to) {
        x.store_partial(size, to);
    }

    constexpr static bool reduce_is_valid() {
        return false;
    }

    static void reduce(Vec8d x, double*to) {
    }

    static void reduce(Vec8d x0, Vec8d x1, Vec8d x2, Vec8d& y) {
    }
};



int main() {
    const int N = 35128;
    double x[N];
    double y[N];
    for (int i = 0; i < N; ++i)
        x[i] = i / 2000.0;

    unroller<agner_exp>(x, y, N);

    auto idx = rand() % N;
    std::cout << "e^" << x[idx] << " = " << y[idx] << std::endl;
}