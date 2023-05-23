
#include <experimental/simd>

namespace stdx = std::experimental;

template<class FUNC, typename IN_T, typename OUT_T>
inline void unroller(IN_T* x, OUT_T* y, const unsigned int n) {
    constexpr unsigned int lane_sz = sizeof(stdx::native_simd<IN_T>) / sizeof(IN_T);

#ifndef __AVX512F__
// entering this section of code is sometimes destructive to performance, and
// sometimes not. this is used when the input size of the data you're operating 
// on is less than the size of the SIMD register and when masked load/store isn't
// available (pre-avx512). in this case, we need to create a vector the size of
// a SIMD register, and move the contents of x and y into it. then do the SIMD
// function processing, and extract the results just from the relevant indexes.
   if( n < lane_sz ) {
        IN_T xa[lane_sz];
        OUT_T ya[lane_sz];
 
        for(unsigned int i = 0; i < n; ++i)
            xa[i] = x[i];
        
        stdx::native_simd<IN_T> xx(xa, stdx::element_aligned);
        stdx::native_simd<IN_T> yy(ya, stdx::element_aligned);
        FUNC::func(xx, yy);
        FUNC::store(yy, ya);
 
        // bug in reduce here. 
        for(unsigned int i = 0; i < n; ++i)
            y[i] = ya[i];
 
        FUNC::reduce(yy, y);
        return;
   }
#endif

    stdx::native_simd<IN_T> xx;
    auto yy = FUNC::y_init();

    unsigned int i = 0;

    if(n > 4*lane_sz) {
        stdx::native_simd<IN_T> xx1, xx2, xx3;
        auto yy1 = FUNC::y_init();
        auto yy2 = FUNC::y_init();
        auto yy3 = FUNC::y_init();

        while(i + 4*lane_sz - 1 < n) {
            xx.copy_from(x + i, stdx::element_aligned);
            i += lane_sz;
            xx1.copy_from(x + i, stdx::element_aligned);
            i += lane_sz;
            xx2.copy_from(x + i, stdx::element_aligned);
            i += lane_sz;
            xx3.copy_from(x + i, stdx::element_aligned);
            i -= 3*lane_sz;

            FUNC::func(xx, yy);
            FUNC::func(xx1, yy1);
            FUNC::func(xx2, yy2);
            FUNC::func(xx3, yy3);

            yy.copy_to(y + i, stdx::element_aligned);
            i += lane_sz;
            yy1.copy_to(y + i, stdx::element_aligned);
            i += lane_sz;
            yy2.copy_to(y + i, stdx::element_aligned);
            i += lane_sz;
            yy3.copy_to(y + i, stdx::element_aligned);
            i += lane_sz;
        }

        FUNC::reduce(yy3, yy2, yy1, yy);
    }

    while (i+lane_sz-1 < n) {
        xx.copy_from(x + i, stdx::element_aligned);
        FUNC::func(xx, yy);
        yy.copy_to(y + i, stdx::element_aligned);
        i += lane_sz;
    }

// these clean up the regions of the array processing that aren't aligned with the lane
// size. if we have avx512, we just mask out the values that exceed the array's allocation, 
// but if we don't it's less clean.
#ifdef __AVX512F__
    // don't know how to do masked loading for std::simd.

    // if(i != n) {
    //     xx = FUNC::maskload(x + i, n-i);
    //     FUNC::maskfunc(xx, n-i, yy);
    //     FUNC::maskstore(yy, n-i, y + i);
    // }

    // FUNC::reduce(yy, y);
#else
    if(i != n) {
        i = n - lane_sz;
        xx = FUNC::load(x + i);
        FUNC::func(xx, yy);
        FUNC::store(yy, y + i);
    } 
#endif
}