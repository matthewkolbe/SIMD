
// noting yet, but i'm thinking

#ifndef H_STD_SIMD_PROPOSED_UNROLLER
#include "std_simd_proposed_unroller.hh"
#define H_STD_SIMD_PROPOSED_UNROLLER
#endif

template<typename T>
class simd_container {
    T * m_values;
    uint64_t m_size;
public:
    simd_container<T>(uint64_t n) : m_size(n) {
        m_values =  new (std::align_val_t(64)) T[m_size];
    }
    ~simd_container<T> () {
        ::operator delete[] (m_values, std::align_val_t(64));
    }
    
    template<class FUNC, typename OUT_T>
    inline void process(FUNC& f, OUT_T* y) {
        unroller(f, m_values, y, m_size);
    }
};

template<typename T>
class simd_view {
    T * m_values;
    int m_size;
public:
    simd_view<T>(T* head, int n) : m_size(n) {
        m_values = head;
    }
    ~simd_view<T> () {
    }
    
    template<class FUNC, typename OUT_T>
    inline void process(FUNC& f, OUT_T* y) {
        unroller(f, m_values, y, m_size);
    }
};