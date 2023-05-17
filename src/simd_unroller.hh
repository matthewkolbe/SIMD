#include <immintrin.h>
#include <iostream>

template<typename IN_T, typename OUT_T, typename INVEC_T, typename OUTVEC_T, class FUNC>
inline void unroller(IN_T* x, OUT_T* y, const unsigned int n) {
    constexpr unsigned int lane_sz = sizeof(INVEC_T) / sizeof(IN_T);

#ifdef __AVX512F__
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
        
        INVEC_T xx = FUNC::load(xa);
        OUTVEC_T yy = FUNC::load(ya);
        FUNC::func(xx, yy);
        FUNC::store(yy, ya);
 
        for(unsigned int i = 0; i < n; ++i)
            y[i] = ya[i];
 
        FUNC::reduce(yy, y);
        return;
   }
#endif

    INVEC_T xx;
    OUTVEC_T yy;
    unsigned int i = 0;

    if(n > 4*lane_sz) {
        INVEC_T xx1, xx2, xx3;
        OUTVEC_T yy1, yy2, yy3;

        while(i + 4*lane_sz - 1 < n) {
            xx = FUNC::load(x + i);
            yy = FUNC::load(y + i);
            i += lane_sz;
            xx1 = FUNC::load(x + i);
            yy1 = FUNC::load(y + i);
            i += lane_sz;
            xx2 = FUNC::load(x + i);
            yy2 = FUNC::load(y + i);
            i += lane_sz;
            xx3 = FUNC::load(x + i);
            yy3 = FUNC::load(y + i);
            i -= 3*lane_sz;

            FUNC::func(xx, yy);
            FUNC::func(xx1, yy1);
            FUNC::func(xx2, yy2);
            FUNC::func(xx3, yy3);

            FUNC::store(yy, y + i);
            i += lane_sz;
            FUNC::store(yy1, y + i);
            i += lane_sz;
            FUNC::store(yy2, y + i);
            i += lane_sz;
            FUNC::store(yy3, y + i);
            i += lane_sz;
        }

        FUNC::reduce(yy3, yy2, yy1, yy);
    }

    while (i+lane_sz-1 < n) {
        xx = FUNC::load(x + i);
        yy = FUNC::load(y + i);
        FUNC::func(xx, yy);
        FUNC::store(yy, y + i);
        i += lane_sz;
    }

// these clean up the regions of the array processing that aren't aligned with the lane
// size. if we have avx512, we just mask out the values that exceed the array's allocation, 
// but if we don't it's less clean.
#ifdef __AVX512F__
    if(i != n) {
        xx = FUNC::maskload(x + i, n-i);
        yy = FUNC::maskload(y + i, n-i);
        FUNC::maskfunc(xx, n-i, yy);
        FUNC::maskstore(yy, n-i, y + i);
    }

    FUNC::reduce(yy, y);
#else
    if(i != n && (!FUNC::reduce_is_valid())) {
        i = n - lane_sz;
        xx = FUNC::load(x + i);
        yy = FUNC::load(y + i);
        FUNC::func(xx, yy);
        FUNC::store(yy, y + i);
    } else if (i != n) {
        // todo: reducing is broken in this path, and i'm not sure what to do.
        
    }
#endif
}