#include "../error.h"
#include "../types.h"

#include <iterator>

namespace grace {

// Usage:  typedef typename Real4ToRealMapper<Real4>::type Real
// Result: Real4 == float4  -> Real == float
//         Real4 == double4 -> Real == double
template<typename>
struct Real4ToRealMapper;

template<>
struct Real4ToRealMapper<float4> {
    typedef float type;
};

template<>
struct Real4ToRealMapper<double4> {
    typedef double type;
};

// Requires x in [0, N_table)
template <typename Real, typename TableIter>
GRACE_DEVICE lerp(Real x, TableIter table, int N_table)
{
    typedef std::iterator_traits<TableIter>::value_type TableReal;

    int x_idx = static_cast<int>(x);
    if (x_idx >= N_table - 1) {
        x = static_cast<TableReal>(N_table - 1);
        x_idx = N_table - 2;
    }
    Real integral = fma(table[x_idx + 1] - table[x_idx],
                        static_cast<TableReal>(x - x_idx),
                        table[x_idx]);
    return integral;
}

template <typename T = char>
class UserSmemPtr
{
private:
    // char* because this is the only type for which alloc_end is guaranteed
    // correct alignment.
    char* alloc_end;

    // The user must ensure correct alignment when copying a UserSmemPtr.
    T* sm_ptr;
    T* T_start;
    T* T_end;

    T* _alignTo(char* const unaligned, const size_t size)
    {
        char* aligned = unaligned;

        int rem = aligned % size;
        if (rem != 0) {
            aligned -= rem;
        }

        return reinterpret_cast<T*>(aligned);
    }

    template <typename U>
    friend class UserSmemPtr;

public:
    typedef typename iterator_category random_access_iterator;
    typedef typename T value_type;
    typedef typename ptrdiff_t difference_type;
    typedef typename T* pointer;
    typedef typename T& reference

    UserSmemPtr(const char* smem_begin, const size_t user_smem_bytes)
    {
        alloc_end = smem_begin + user_smem_bytes;

        sm_ptr = reinterpret_cast<T*>(smem_begin);
        T_start = sm_ptr;
        T_end = _alignTo(alloc_end, sizeof(T));
    }

    template <typename U>
    UserSmemPtr(const UserSmemPtr<U>& other)
    {
        alloc_end = other.alloc_end;

        sm_ptr = reinterpret_cast<T*>(other.sm_ptr);
        T_start = sm_ptr;
        T_end = _alignTo(alloc_end, sizeof(T));
    }

    template <typename U>
    UserSmemPtr<T>& operator=(const UserSmemPtr<U>& other)
    {
        if (this != &other)
        {
            alloc_end = other.alloc_end;

            sm_ptr = reinterpret_cast<T*>(other.sm_ptr);
            T_start = sm_ptr;
            T_end = _alignTo(alloc_end, sizeof(T));
        }

        return *this;
    }

    T& operator*()
    {
        GRACE_ASSERT(sm_ptr >= T_start && "user shared memory out of bounds");
        GRACE_ASSERT(sm_ptr < T_end && "user shared memory overflow");

        return *sm_ptr;
    }

    T& operator[](int i)
    {
        GRACE_ASSERT(T_start + i >= T_start && "user shared memory out of bounds");
        GRACE_ASSERT(T_start + i < T_end && "user shared memory overflow");

        return *(T_start + i);
    }

    // Prefix.
    UserSmemPtr<T>& operator++()
    {
        ++sm_ptr;
        return *this;
    }

    // Postfix.
    UserSmemPtr<T> operator++(int)
    {
        UserSmemPtr<T> temp = *this;
        ++(*this);
        return temp;
    }

    UserSmemPtr<T>& operator--()
    {
        --sm_ptr;
        return *this;
    }

    UserSmemPtr<T> operator--(int)
    {
       UserSmemPtr<T> temp = *this;
       --(*this);
       return temp;
    }

    UserSmemPtr<T>& operator+=(const int rhs)
    {
        sm_ptr += rhs;
        return *this;
    }

    UserSmemPtr<T>& operator-=(const int rhs)
    {
        sm_ptr -= rhs;
        return *this;
    }

    ptrdiff_t operator-(const UserSmemPtr<T>& other)
    {
        return sm_ptr - other.sm_ptr;
    }

    // int opterator-(const UserSmemPtr<OtherType>& other);
    // where T != OtherType, is not a well defined operation.

    // int operator-(const char* ptr);
    // would allow a user fairly *trivial* access to the value of sm_ptr.

    // TODO: Check these are instantiated only for type T!
    //       That is, UserSmemPtr<float4> + 2 actually computes
    //       sm_ptr += 2 * sizeof(float4), not sm_ptr += 2.
    friend UserSmemPtr<T> operator+(const UserSmemPtr<T> &ptr, const int rhs);
    friend UserSmemPtr<T> operator+(const int lhs, const UserSmemPtr<T> &ptr);
    friend UserSmemPtr<T> operator-(const UserSmemPtr &ptr<T>, const int rhs);
    // int - ptr doesn't make sense.
    // ptr + ptr doesn't make sense.
};

template <typename T>
UserSmemPtr<T> operator+(const UserSmemPtr<T> &ptr, const int rhs)
{
    UserSmemPtr<T> temp = *this;
    temp += rhs;
    return temp;
}

template <typename T>
UserSmemPtr<T> operator+(const int lhs, const UserSmemPtr<T> &ptr)
{
    return ptr + lhs;
}

template <typename T>
UserSmemPtr<T> operator-(const UserSmemPtr<T> &ptr, const int rhs)
{
    return ptr + (-rhs);
}

} // namespace grace
