#pragma once

#include "meta.h"

#include "../error.h"
#include "../types.h"

#include <iterator>

namespace grace {

namespace gpu {

template <typename T>
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

    GRACE_DEVICE T* _alignTo(char* const unaligned, const size_t size)
    {
        char* aligned = unaligned;

        int rem = (uintptr_t)aligned % size;
        if (rem != 0) {
            aligned -= rem;
        }

        return reinterpret_cast<T*>(aligned);
    }

    // So copy/assignment constructors can accept UserSmemPtrs of a different
    // type from T.
    template <typename U>
    friend class UserSmemPtr;

public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef T value_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;

    GRACE_DEVICE UserSmemPtr(char* const smem_begin,
                             const size_t user_smem_bytes)
    {
        alloc_end = smem_begin + user_smem_bytes;

        sm_ptr = reinterpret_cast<T*>(smem_begin);
        T_start = sm_ptr;
        T_end = _alignTo(alloc_end, sizeof(T));
    }

    template <typename U>
    GRACE_DEVICE UserSmemPtr(const UserSmemPtr<U>& other)
    {
        alloc_end = other.alloc_end;

        sm_ptr = reinterpret_cast<T*>(other.sm_ptr);
        T_start = sm_ptr;
        T_end = _alignTo(alloc_end, sizeof(T));
    }

    template <typename U>
    GRACE_DEVICE UserSmemPtr<T>& operator=(const UserSmemPtr<U>& other)
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

    GRACE_DEVICE T& operator*()
    {
        GRACE_ASSERT(sm_ptr >= T_start && "user shared memory out of bounds");
        GRACE_ASSERT(sm_ptr < T_end && "user shared memory overflow");

        return *sm_ptr;
    }

    GRACE_DEVICE T& operator[](int i)
    {
        GRACE_ASSERT(T_start + i >= T_start && "user shared memory out of bounds");
        GRACE_ASSERT(T_start + i < T_end && "user shared memory overflow");

        return *(T_start + i);
    }

    // Prefix.
    GRACE_DEVICE UserSmemPtr<T>& operator++()
    {
        ++sm_ptr;
        return *this;
    }

    // Postfix.
    GRACE_DEVICE UserSmemPtr<T> operator++(int)
    {
        UserSmemPtr<T> temp = *this;
        ++(*this);
        return temp;
    }

    GRACE_DEVICE UserSmemPtr<T>& operator--()
    {
        --sm_ptr;
        return *this;
    }

    GRACE_DEVICE UserSmemPtr<T> operator--(int)
    {
       UserSmemPtr<T> temp = *this;
       --(*this);
       return temp;
    }

    GRACE_DEVICE UserSmemPtr<T>& operator+=(const int rhs)
    {
        sm_ptr += rhs;
        return *this;
    }

    GRACE_DEVICE UserSmemPtr<T>& operator-=(const int rhs)
    {
        sm_ptr -= rhs;
        return *this;
    }

    GRACE_DEVICE ptrdiff_t operator-(const UserSmemPtr<T>& other)
    {
        return sm_ptr - other.sm_ptr;
    }

    // int opterator-(const UserSmemPtr<OtherType>& other);
    // where T != OtherType, is not a well defined operation.

    // int operator-(const char* ptr);
    // would allow a user fairly *trivial* access to the value of sm_ptr.

    GRACE_DEVICE friend UserSmemPtr<T> operator+(const UserSmemPtr<T> ptr,
                                                 const int rhs)
    {
        ptr += rhs;
        return ptr;
    }

    GRACE_DEVICE friend UserSmemPtr<T> operator+(const int lhs,
                                                 const UserSmemPtr<T>& ptr)
    {
        return ptr + lhs;
    }

    GRACE_DEVICE friend UserSmemPtr<T> operator-(const UserSmemPtr<T>& ptr,
                                                 const int rhs)
    {
        return ptr + (-rhs);
    }
    // int - ptr doesn't make sense.
    // ptr + ptr doesn't make sense.
};

} // namespace gpu

} // namespace grace
