#include <vector>
#include <iostream>


template<class T> 
class simd {
    friend class simd<bool>;
    friend class simd<double>;
    friend class simd<float>;
    friend class simd<int>;
    friend class simd<char>;
    friend class simd_array<bool>;
    friend class simd_array<double>;
    friend class simd_array<float>;
    friend class simd_array<int>;
    friend class simd_array<char>;

    T m_value;

public:
    simd<T> () : m_value(T{}){}
    simd<T> (T v) : m_value(v){}

    friend std::ostream& operator<<(std::ostream& os, const simd<T>& o)
    {
        os << o.m_value;
        return os;
    }

    constexpr simd<T> operator+(const simd<T>& a) {
         return simd<T>(a.m_value + m_value);  }

    constexpr simd<T> operator-(const simd<T>& a) {
         return simd<T>(m_value - a.m_value); }

    constexpr simd<T> operator*(const simd<T>& a) {
         return simd<T>(m_value * a.m_value); }

    constexpr simd<T> operator+=(const simd<T>& a) {
        m_value += a.m_value;
        return *this;  }

    simd<bool> operator==(const simd<T>& a) {
         return simd<bool>(m_value == a.m_value); }
        
    simd<bool> operator>(const simd<T>& a) {
         return simd<bool>(m_value > a.m_value); }

    simd<bool> operator<(const simd<T>& a) {
         return simd<bool>(m_value < a.m_value); }

    simd<bool> operator>=(const simd<T>& a) {
         return simd<bool>(m_value >= a.m_value); }

    simd<bool> operator<=(const simd<T>& a) {
         return simd<bool>(m_value <= a.m_value); }

    simd<bool> operator&&(const simd<bool>& a) {
         return simd<bool>(m_value && a.m_value); }

    simd<bool> operator||(const simd<bool>& a) {
         return simd<bool>(m_value || a.m_value); }

    constexpr void ifsimd(const simd<bool>& condition, const simd<T>& if_eq) {
        if(condition.m_value)
            (*this) = if_eq;
    }

    constexpr simd<T> ifsimd(const simd<bool>& condition, const simd<T>& if_eq, const simd<T>& if_neq) {
        simd<T> r = if_eq;
        if(!condition.m_value)
            r = if_neq;
        return r;
    }

    constexpr void ifeq(const simd<T>& condition, const simd<T>& if_eq) {
        ifsimd(*this == condition, if_eq);
    }

    constexpr simd<T> ifeq(const simd<T>& condition, const simd<T>& if_eq, const simd<T>& if_neq) {
        return ifsimd(*this == condition, if_eq, if_neq);
    }

    constexpr void ifgt(const simd<T>& condition, const simd<T>& if_gt) {
        ifsimd(*this > condition, if_gt);
    }

    constexpr simd<T> ifgt(const simd<T>& condition, const simd<T>& if_gt, const simd<T>& if_ngt) {
        return ifsimd(*this > condition, if_gt, if_ngt);
    }

    template<typename F>
    static void repeat(simd<int> to, F && inner) {
        #ifdef __clang__
            #pragma clang loop vectorize(enable)
        #elif __GNUC__
            #pragma GCC ivdep
        #endif
        for(int i = 0; i < to.m_value; ++i)
            inner(i);
    }
};

template<class T> 
class simd_array {
    simd<T>* m_values;
    simd<int> m_size;
public:
    simd_array<T>(const T * seed, int n): m_size(n) {
        m_values =  = new (std::align_val_t(64)) simd<T>[m_size];
        simd<T>::repeat(m_size, [&](int i) {
            m_values[i] = seed[i];
        });
    }
    simd_array<T>(int n) : m_size(n) {
        m_values =  = new (std::align_val_t(64)) simd<T>[m_size];
    }
    ~simd_array<T> () {
        ::operator delete[] (m_values, std::align_val_t(64));
    }

    simd<bool> operator==(const simd_array<T>& a) {

        simd<bool> rr{true};
        simd<T>::repeat(m_size, [&](simd<int> i) {
            rr |= m_values[i.m_value] == a.at(i);
        });
        return rr; 
    }

    simd<int> get_length() { return m_size; }

    simd<T> operator [](int i) const    {return m_values[i];}
    simd<T> & operator [](int i) {return m_values[i];}
    simd<T>& at(int i ) {return m_values[i];}
    simd<T>& at(simd<int> i ) {return m_values[i.m_value];}

    template<typename F>
    void repeat(simd<int> to, F && inner) {
        repeat(0, to, inner);
    }

    template<typename F>
    void repeat(simd<int> from, simd<int> to, F && inner) {
        #ifdef __clang__
            #pragma clang loop vectorize(enable)
        #elif __GNUC__
            #pragma GCC ivdep
        #endif
        for(int i = from.m_value; i < to.m_value; ++i)
            inner(i, m_values.at(i));
    }
};


// David's example
class rgba {
    simd<int> m_r, m_g, m_b, m_a;

public:
    rgba(){}
    rgba(int r, int g, int b, int a) : m_r(r), m_g(g), m_b(b), m_a(a) {}

    void set_all(int r, int g, int b, int a) {
        m_r = r;
        m_g = g;
        m_b = b;
        m_a = a;
    }

    friend std::ostream& operator<<(std::ostream& os, const rgba& o)
    {
        os << o.m_r << " " << o.m_g << " " << o.m_b << " " << o.m_a;
        return os;
    }

    void zero_if_equal(const rgba & v) {
        auto eqr = m_r == v.m_r;
        auto eqg = m_g == v.m_g;
        auto eqb = m_b == v.m_b;
        auto eqa = m_a == v.m_a;
        auto eq = eqr && eqg && eqb && eqa;
        simd<int> zero{0};
        m_r.ifsimd(eq, zero);
        m_g.ifsimd(eq, zero);
        m_b.ifsimd(eq, zero);
        m_a.ifsimd(eq, zero);
    }

    static void zero_if_equal_vector(std::vector<rgba> & pixels, const rgba & v) {
        for(auto& pixel: pixels)
            pixel.zero_if_equal(v);
    }

    // vectorizes only at the rgba class level, but should be able to vectorize at the 4x class level,
    // but also the inner loop is so clean it may be that 4x vectorization is bad?
    static void zero_if_equal_vector(rgba * pixels, int n, const rgba & v) {
        for(int i = 0; i < n; ++i)
            pixels[i].zero_if_equal(v);
    }

    void brighten_if_darker_than(const rgba & v, const simd<int> & lightener) {
        auto lighter = m_a - lightener;
        m_a.ifgt(v.m_a, lighter);
    }

    // vectorizes at the 4x class level
    static void brighten_if_darker_than_vector(std::vector<rgba> & pixels, const rgba & v, const simd<int> & lightener) {
        for(auto& pixel: pixels)
            pixel.brighten_if_darker_than(v, lightener);
    }

    static void brighten_if_darker_than_vector(rgba* pixels, int n, const rgba & v, const simd<int> & lightener) {
        simd<int>::repeat(n, [&](int i) {
            pixels[i].brighten_if_darker_than(v, lightener);
        });
        //for(int i = 0; i < n; ++i)
        //    pixels[i].brighten_if_darker_than(v, lightener);
    }
};

// HPC example
class vec {
    simd_array<double>* m_vec;
public:
    vec(int n) {
        m_vec = new simd_array<double> (n);
    }
    ~vec() {
        delete[] m_vec;
    }

    void set(int i, double v) {
        m_vec[i] = v;
    }

    simd<double> dot(const vec & a) {
        simd<double> r{0.0};
        m_vec->repeat(m_vec->get_length(), [&](int i, simd<double>& v){
            r += v * a.m_vec->at(i);
        });
        return r;
    }
};

// Substring find
class substrfind {
    simd_array<char> m_vec;
    simd<int> end{0};
public:
    substrfind(int n) {
        m_vec = simd_array<char> (n);
    }

    void insert(simd_array<char>& v) {
        for(int i = end; i < end + insert.length(); ++i)
            m_vec[i] = insert[i - end];
        end += insert.length();
        m_vec[end] = '/n';
        end++;
    }

    int find_substr(simd_array<char> & sub) {
        simd<char>::repeat(end - sub.get_length(), [&](int i){

        });
    }
    
};


int main() {
    rgba pixels[10000];
    rgba v(100, 100, 100, 100);
    simd<int> light(100);
    for(int i = 0; i < 10000; ++i)
        pixels[i].set_all(i%256, i%256, i%256, i%256);

    rgba::brighten_if_darker_than_vector(pixels, 10000, v, light);
    //rgba::zero_if_equal_vector(pixels, 10000, v);

    std::cout << pixels[256*5+100] << std::endl;

    vec a(1000);
    vec b(1000);

    for(int i = 0; i < 1000; ++i) {
        a.set(i, i/100.0);
        b.set(i, i/1000.0);
    }
    
    auto d = a.dot(b);
    
    std::cout << d << std::endl;
}