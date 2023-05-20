
// noting yet, but i'm thinking

template<typename T>
class simd_container {
    T * m_values;
    uint64_t n;
public:
    simd_container<T>(int n) : m_size(n) {
        m_values =  = new (std::align_val_t(64)) T[m_size];
    }
    ~simd_container<T> () {
        ::operator delete[] (m_values, std::align_val_t(64));
    }
    constexpr simd_container<T> operator+(const simd_container<T>& a) {
        simd_container<T> r(a.n);

        return (a.m_value + m_value);  }

};