

#ifndef H_INTRINSIC_GENERATOR
#include "intrinsic_generator.hh"
#define H_INTRINSIC_GENERATOR
#endif

template <class DERIVED, typename T> 
struct UnrollerUnit
{
    template<typename VEC_T>
    inline auto func(VEC_T x, VEC_T & y) {
        return static_cast<DERIVED*>(this)->func(x, y);
    }

    template<typename VEC_T>
    inline auto maskfunc(VEC_T x, const unsigned int size,  VEC_T & y) {
        return static_cast<DERIVED*>(this)->maskfunc(x, size, y);
    }

    inline auto x_init()
    {
        constexpr bool has_x_init = requires(DERIVED t) {
            t.x_init();
        };

        if constexpr (has_x_init)
            return static_cast<DERIVED*>(this)->x_init();
        else
            return intr::set1<T>()(0); 
    }

    inline auto y_init()
    {
        constexpr bool has_y_init = requires(DERIVED t) {
            t.y_init();
        };

        if constexpr (has_y_init)
            return static_cast<DERIVED*>(this)->y_init();
        else
            return intr::set1<T>()(0); 
    }

    template<typename VEC_T>
    inline auto store(T*to, VEC_T x) 
    {
        constexpr bool has_store = requires(DERIVED t) {
            t.store(x, to);
        };

        if constexpr (has_store)
            static_cast<DERIVED*>(this)->store(to, x);
        else
            intr::storeu<T>()(to, x);
    }

    template<typename VEC_T>
    inline auto maskstore(T*to, VEC_T x, unsigned int places) 
    {
        constexpr bool has_store = requires(DERIVED t) {
            t.maskstore(x, to);
        };

        if constexpr (has_store)
            static_cast<DERIVED*>(this)->store(to, x, places);
        else{
            auto mask = ~(intr::full_mask<T>()() << places);
            intr::mask_storeu<T>()(to, x, mask);
        }
    }

    template<typename VEC_T>
    inline auto reduce(VEC_T x, T*to) 
    {
        constexpr bool has_reduce = requires(DERIVED t) {
            t.reduce(x, to);
        };

        if constexpr (has_reduce)
            static_cast<DERIVED*>(this)->reduce(x, to);
    }

    template<typename VEC_T>
    inline auto reduce_vec(VEC_T x0, VEC_T x1, VEC_T x2, VEC_T& y)
    {
        constexpr bool has_reduce = requires(DERIVED t) {
            t.reduce_vec(x0, x1, x2, y);
        };

        if constexpr (has_reduce)
            static_cast<DERIVED*>(this)->reduce_vec(x0, x1, x2, y);
    }
};


template<class FUNC, typename IN_T, typename OUT_T>
inline void unroller(FUNC& f, IN_T* x, OUT_T* y, const unsigned int n) {
    constexpr unsigned int lane_sz = sizeof(f.x_init()) / sizeof(IN_T);

    auto xx = f.x_init();
    auto yy = f.y_init();

    unsigned int i = 0;

    if(n > 4*lane_sz) {
        auto xx1 = f.x_init();
        auto yy1 = f.y_init();
        auto xx2 = f.x_init();
        auto yy2 = f.y_init();
        auto xx3 = f.x_init();
        auto yy3 = f.y_init();

        while(i + 4*lane_sz - 1 < n) {
            xx = intr::loadu<IN_T>()(x + i);
            i += lane_sz;
            xx1 = intr::loadu<IN_T>()(x + i);
            i += lane_sz;
            xx2 = intr::loadu<IN_T>()(x + i);
            i += lane_sz;
            xx3 = intr::loadu<IN_T>()(x + i);
            i -= 3*lane_sz;

            f.func(xx, yy);
            f.func(xx1, yy1);
            f.func(xx2, yy2);
            f.func(xx3, yy3);

            f.store(y + i, yy);
            i += lane_sz;
            f.store(y + i, yy);
            i += lane_sz;
            f.store(y + i, yy);
            i += lane_sz;
            f.store(y + i, yy);
            i += lane_sz;
        }

        f.reduce_vec(yy3, yy2, yy1, yy);
    }

    while (i+lane_sz-1 < n) {
        xx = intr::loadu<IN_T>()(x + i);
        f.func(xx, yy);
        f.store(y + i, yy);
        i += lane_sz;
    }

// these clean up the regions of the array processing that aren't aligned with the lane
// size. if we have avx512, we just mask out the values that exceed the array's allocation, 
// but if we don't it's less clean.
#ifdef __AVX512F__
    if(i != n) {
        auto x_mask = ~(intr::full_mask<IN_T>()() << (n-i));
        xx = intr::maskz_loadu<IN_T>()(x + i, x_mask);
        f.maskfunc(xx, n-i, yy);
        f.maskstore(y + i, yy, n-i);
    }

    f.reduce(yy, y);
#else
    // if(i != n) {
    //     i = n - lane_sz;
    //     xx = FUNC::load(x, i);
    //     FUNC::func(xx, yy);
    //     FUNC::store(yy, i, y);
    // } 
#endif
}


