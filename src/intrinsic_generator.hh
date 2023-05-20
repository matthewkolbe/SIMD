#include <type_traits>

#define M_TWO_OPERATION(OPERATION) template<typename T, int SZ>\
inline auto constexpr vec_##OPERATION() {\
    if constexpr (std::is_same<T, double   >::value && SZ == 512) return [](__m512d& a, __m512d& b) { return _mm512_##OPERATION##_pd(a, b); };\
    if constexpr (std::is_same<T, float    >::value && SZ == 512) return [](__m512 & a, __m512 & b) { return _mm512_##OPERATION##_ps(a, b); };\
    if constexpr (std::is_same<T, char     >::value && SZ == 512) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi8(a, b); };\
    if constexpr (std::is_same<T, short    >::value && SZ == 512) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi16(a, b); };\
    if constexpr (std::is_same<T, int      >::value && SZ == 512) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi32(a, b); };\
    if constexpr (std::is_same<T, long long>::value && SZ == 512) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi64(a, b); };\
    if constexpr (std::is_same<T, double   >::value && SZ == 256) return [](__m256d& a, __m256d& b) { return _mm256_##OPERATION##_pd(a, b); };\
    if constexpr (std::is_same<T, float    >::value && SZ == 256) return [](__m256 & a, __m256 & b) { return _mm256_##OPERATION##_ps(a, b); };\
    if constexpr (std::is_same<T, char     >::value && SZ == 256) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi8(a, b); };\
    if constexpr (std::is_same<T, short    >::value && SZ == 256) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi16(a, b); };\
    if constexpr (std::is_same<T, int      >::value && SZ == 256) return [](__m512i& a, __m512i& b) { return _mm512_##OPERATION##_epi32(a, b); };\
    if constexpr (std::is_same<T, long long>::value && SZ == 256) return [](__m256i& a, __m256i& b) { return _mm256_##OPERATION##_epi64(a, b); };}\
template<typename T, int SZ>\
inline auto constexpr vec_maskz_##OPERATION() {\
    if constexpr (std::is_same<T, double   >::value && SZ == 512) return [](__m512d& a, __m512d& b, __mmask8  mask) { return _mm512_maskz_##OPERATION##_pd(mask, a, b); };\
    if constexpr (std::is_same<T, float    >::value && SZ == 512) return [](__m512 & a, __m512 & b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_ps(mask, a, b); };\
    if constexpr (std::is_same<T, char     >::value && SZ == 512) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi8(mask, a, b); };\
    if constexpr (std::is_same<T, short    >::value && SZ == 512) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi16(mask, a, b); };\
    if constexpr (std::is_same<T, int      >::value && SZ == 512) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi32(mask, a, b); };\
    if constexpr (std::is_same<T, long long>::value && SZ == 512) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi64(mask, a, b); };\
    if constexpr (std::is_same<T, double   >::value && SZ == 256) return [](__m256d& a, __m256d& b, __mmask8  mask) { return _mm256_maskz_##OPERATION##_pd(mask, a, b); };\
    if constexpr (std::is_same<T, float    >::value && SZ == 256) return [](__m256 & a, __m256 & b, __mmask8  mask) { return _mm256_maskz_##OPERATION##_ps(mask, a, b); };\
    if constexpr (std::is_same<T, char     >::value && SZ == 256) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi8(mask, a, b); };\
    if constexpr (std::is_same<T, short    >::value && SZ == 256) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi16(mask, a, b); };\
    if constexpr (std::is_same<T, int      >::value && SZ == 256) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi32(mask, a, b); };\
    if constexpr (std::is_same<T, long long>::value && SZ == 256) return [](__m512i& a, __m512i& b, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi64(mask, a, b); };}\
template<typename T, int SZ>\
inline auto constexpr vec_mask_##OPERATION() {\
    if constexpr (std::is_same<T, double   >::value && SZ == 512) return [](__m512d& src, __m512d& a, __m512d& b, __mmask8  mask) { return _mm512_mask_##OPERATION##_pd(src, mask, a, b); };\
    if constexpr (std::is_same<T, float    >::value && SZ == 512) return [](__m512 & src, __m512 & a, __m512 & b, __mmask16 mask) { return _mm512_mask_##OPERATION##_ps(src, mask, a, b); };\
    if constexpr (std::is_same<T, char     >::value && SZ == 512) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi8(src, mask, a, b); };\
    if constexpr (std::is_same<T, short    >::value && SZ == 512) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi16(src, mask, a, b); };\
    if constexpr (std::is_same<T, int      >::value && SZ == 512) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi32(src, mask, a, b); };\
    if constexpr (std::is_same<T, long long>::value && SZ == 512) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi64(src, mask, a, b); };\
    if constexpr (std::is_same<T, double   >::value && SZ == 256) return [](__m256d& src, __m256d& a, __m256d& b, __mmask8  mask) { return _mm256_mask_##OPERATION##_pd(src, mask, a, b); };\
    if constexpr (std::is_same<T, float    >::value && SZ == 256) return [](__m256 & src, __m256 & a, __m256 & b, __mmask8  mask) { return _mm256_mask_##OPERATION##_ps(src, mask, a, b); };\
    if constexpr (std::is_same<T, char     >::value && SZ == 256) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi8(src, mask, a, b); };\
    if constexpr (std::is_same<T, short    >::value && SZ == 256) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi16(src, mask, a, b); };\
    if constexpr (std::is_same<T, int      >::value && SZ == 256) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi32(src, mask, a, b); };\
    if constexpr (std::is_same<T, long long>::value && SZ == 256) return [](__m512i& src, __m512i& a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi64(src, mask, a, b); };}\


#define M_LOAD(OPERATION) template<typename T, int SZ>\
inline auto constexpr vec_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](T* a) { return _mm512_##OPERATION##_pd(a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](T* a) { return _mm512_##OPERATION##_ps(a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](T* a) { return _mm512_##OPERATION##_epi32(a); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](T* a) { return _mm256_##OPERATION##_pd(a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](T* a) { return _mm256_##OPERATION##_ps(a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](T* a) { return _mm256_##OPERATION##_epi32(a); };}\
template<typename T, int SZ>\
inline auto constexpr vec_maskz_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](T* a, __mmask8  mask) { return _mm512_maskz_##OPERATION##_pd(mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](T* a, __mmask16 mask) { return _mm512_maskz_##OPERATION##_ps(mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](T* a, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi32(mask, a); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](T* a, __mmask8  mask) { return _mm256_maskz_##OPERATION##_pd(mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](T* a, __mmask8  mask) { return _mm256_maskz_##OPERATION##_ps(mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](T* a, __mmask8  mask) { return _mm256_maskz_##OPERATION##_epi32(mask, a); };}\
template<typename T, int SZ>\
inline auto constexpr vec_mask_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](__m512d& src, T* a, __mmask8  mask) { return _mm512_mask_##OPERATION##_pd(src, mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](__m512 & src, T* a, __mmask16 mask) { return _mm512_mask_##OPERATION##_ps(src, mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](__m512i& src, T* a, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi32(src, mask, a); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](__m256d& src, T* a, __mmask8  mask) { return _mm256_mask_##OPERATION##_pd(src, mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](__m256 & src, T* a, __mmask8  mask) { return _mm256_mask_##OPERATION##_ps(src, mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](__m256i& src, T* a, __mmask8  mask) { return _mm256_mask_##OPERATION##_epi32(src, mask, a); };}\



#define M_STORE(OPERATION) template<typename T, int SZ>\
inline auto constexpr vec_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](T* a, __m512d& b) { return _mm512_##OPERATION##_pd(a, b); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](T* a, __m512 & b) { return _mm512_##OPERATION##_ps(a, b); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](T* a, __m512i& b) { return _mm512_##OPERATION##_epi32(a, b); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](T* a, __m256d& b) { return _mm256_##OPERATION##_pd(a, b); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](T* a, __m256 & b) { return _mm256_##OPERATION##_ps(a, b); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](T* a, __m256i& b) { return _mm256_##OPERATION##_epi32(a, b); };}\
template<typename T, int SZ>\
inline auto constexpr vec_mask_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](T* a, __m512d& b, __mmask8  mask) { return _mm512_mask_##OPERATION##_pd(a, mask, b); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](T* a, __m512 & b, __mmask16 mask) { return _mm512_mask_##OPERATION##_ps(a, mask, b); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](T* a, __m512i& b, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi32(a, mask, b); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](T* a, __m256d& b, __mmask8  mask) { return _mm256_mask_##OPERATION##_pd(a, mask, b); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](T* a, __m256 & b, __mmask8  mask) { return _mm256_mask_##OPERATION##_ps(a, mask, b); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](T* a, __m256i& b, __mmask8  mask) { return _mm256_mask_##OPERATION##_epi32(a, mask, b); };}\


#define M_SET(OPERATION) template<typename T, int SZ>\
inline auto constexpr vec_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](T a) { return _mm512_##OPERATION##_pd(a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](T a) { return _mm512_##OPERATION##_ps(a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](T a) { return _mm512_##OPERATION##_epi32(a); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](T a) { return _mm256_##OPERATION##_pd(a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](T a) { return _mm256_##OPERATION##_ps(a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](T a) { return _mm256_##OPERATION##_epi32(a); };}\
template<typename T, int SZ>\
inline auto constexpr vec_maskz_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](T a, __mmask8  mask) { return _mm512_maskz_##OPERATION##_pd(mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](T a, __mmask16 mask) { return _mm512_maskz_##OPERATION##_ps(mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](T a, __mmask16 mask) { return _mm512_maskz_##OPERATION##_epi32(mask, a); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](T a, __mmask8  mask) { return _mm256_maskz_##OPERATION##_pd(mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](T a, __mmask8  mask) { return _mm256_maskz_##OPERATION##_ps(mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](T a, __mmask8  mask) { return _mm256_maskz_##OPERATION##_epi32(mask, a); };}\
template<typename T, int SZ>\
inline auto constexpr vec_mask_##OPERATION() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return [](__m512d& src, T a, __mmask8  mask) { return _mm512_mask_##OPERATION##_pd(src, mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return [](__m512 & src, T a, __mmask16 mask) { return _mm512_mask_##OPERATION##_ps(src, mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return [](__m512i& src, T a, __mmask16 mask) { return _mm512_mask_##OPERATION##_epi32(src, mask, a); };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return [](__m256d& src, T a, __mmask8  mask) { return _mm256_mask_##OPERATION##_pd(src, mask, a); };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return [](__m256 & src, T a, __mmask8  mask) { return _mm256_mask_##OPERATION##_ps(src, mask, a); };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return [](__m256i& src, T a, __mmask8  mask) { return _mm256_mask_##OPERATION##_epi32(src, mask, a); };}\

#define M_FULL_MASK() template<typename T, int SZ>\
inline auto constexpr vec_full_mask() {\
    if constexpr (std::is_same<T, double>::value && SZ == 512) return []() { return (__mmask8)0xFF; };\
    if constexpr (std::is_same<T, float >::value && SZ == 512) return []() { return (__mmask16)0xFFFF;  };\
    if constexpr (std::is_same<T, int   >::value && SZ == 512) return []() { return (__mmask16)0xFFFF; };\
    if constexpr (std::is_same<T, double>::value && SZ == 256) return []() { return (__mmask8)0xFF; };\
    if constexpr (std::is_same<T, float >::value && SZ == 256) return []() { return (__mmask8)0xFF; };\
    if constexpr (std::is_same<T, int   >::value && SZ == 256) return []() { return (__mmask8)0xFF; };}\


M_TWO_OPERATION(add)

M_TWO_OPERATION(mul)

M_TWO_OPERATION(sub)

M_LOAD(loadu)
M_LOAD(load)

M_STORE(storeu)
M_STORE(store)

M_SET(set1)

M_FULL_MASK()