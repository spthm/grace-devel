#pragma once

#include "meta.h"

#include "../error.h"
#include "../types.h"

#include <cstddef>
#include <iterator>

namespace grace {

namespace gpu {

// An iterator which may only access the underlying array in a specific range.
// Other than that, no constraints are placed on the type accessed, and user's
// must therefore take care to ensure correct alignment.
template <typename T>
class BoundIter
{
private:
    // char* because this is the only type for which alloc_end is guaranteed
    // correct alignment.
    char* alloc_end;

    // The user must ensure correct alignment when copying a BoundIter.
    T* ptr;
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

    // So copy/assignment constructors can accept BoundIters of a different
    // type from T.
    template <typename U>
    friend class BoundIter;

public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef T value_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;

    GRACE_DEVICE BoundIter(char* const begin, const size_t bytes)
    {
        alloc_end = begin + bytes;

        ptr = reinterpret_cast<T*>(begin);
        T_start = ptr;
        T_end = _alignTo(alloc_end, sizeof(T));
    }

    template <typename U>
    GRACE_DEVICE BoundIter(const BoundIter<U>& other)
    {
        alloc_end = other.alloc_end;

        ptr = reinterpret_cast<T*>(other.ptr);
        T_start = ptr;
        T_end = _alignTo(alloc_end, sizeof(T));
    }

    template <typename U>
    GRACE_DEVICE BoundIter<T>& operator=(const BoundIter<U>& other)
    {
        if (this != &other)
        {
            alloc_end = other.alloc_end;

            ptr = reinterpret_cast<T*>(other.ptr);
            T_start = ptr;
            T_end = _alignTo(alloc_end, sizeof(T));
        }

        return *this;
    }

    GRACE_DEVICE T& operator*()
    {
        GRACE_ASSERT(ptr >= T_start && "user shared memory out of bounds");
        GRACE_ASSERT(ptr < T_end && "user shared memory overflow");

        return *ptr;
    }

    GRACE_DEVICE const T& operator*() const
    {
        GRACE_ASSERT(ptr >= T_start && "user shared memory out of bounds");
        GRACE_ASSERT(ptr < T_end && "user shared memory overflow");

        return *ptr;
    }

    GRACE_DEVICE T& operator[](difference_type i)
    {
        return *((*this) + i);
    }

    GRACE_DEVICE const T& operator[](difference_type i) const
    {
        return *((*this) + i);
    }

    GRACE_DEVICE T* operator->()
    {
        return &(*(*this));
    }

    GRACE_DEVICE const T* operator->() const
    {
        return &(*(*this));
    }

    // Prefix.
    GRACE_DEVICE BoundIter<T>& operator++()
    {
        ++ptr;
        return *this;
    }

    // Postfix.
    GRACE_DEVICE BoundIter<T> operator++(int)
    {
        BoundIter<T> temp = *this;
        ++(*this);
        return temp;
    }

    // Prefix.
    GRACE_DEVICE BoundIter<T>& operator--()
    {
        --ptr;
        return *this;
    }

    // Postfix.
    GRACE_DEVICE BoundIter<T> operator--(int)
    {
       BoundIter<T> temp = *this;
       --(*this);
       return temp;
    }

    GRACE_DEVICE BoundIter<T>& operator+=(const difference_type n)
    {
        ptr += n;
        return *this;
    }

    GRACE_DEVICE BoundIter<T>& operator-=(const difference_type n)
    {
        ptr -= n;
        return *this;
    }

    GRACE_DEVICE friend BoundIter<T> operator+(const BoundIter<T>& lhs,
                                               const difference_type rhs)
    {
        BoundIter<T> temp = lhs;
        temp += rhs;
        return temp;
    }

    GRACE_DEVICE friend BoundIter<T> operator+(const difference_type lhs,
                                               const BoundIter<T>& rhs)
    {
        // Swap sides.
        return rhs + lhs;
    }

    GRACE_DEVICE friend BoundIter<T> operator-(const BoundIter<T>& lhs,
                                               const difference_type rhs)
    {
        return lhs + (-rhs);
    }

    GRACE_DEVICE difference_type operator-(const BoundIter<T>& other) const
    {
        return ptr - other.ptr;
    }

    GRACE_DEVICE friend bool operator==(const BoundIter<T>& lhs,
                                        const BoundIter<T>& rhs)
    {
        return lhs.ptr == rhs.ptr;
    }

    GRACE_DEVICE friend bool operator!=(const BoundIter<T>& lhs,
                                        const BoundIter<T>& rhs)
    {
        return !(lhs == rhs);
    }

    GRACE_DEVICE friend bool operator<(const BoundIter<T>& lhs,
                                       const BoundIter<T>& rhs)
    {
        return (rhs - lhs) > 0;
    }

    GRACE_DEVICE friend bool operator>(const BoundIter<T>& lhs,
                                       const BoundIter<T>& rhs)
    {
        // Swap sides.
        return rhs < lhs;
    }

    GRACE_DEVICE friend bool operator<=(const BoundIter<T>& lhs,
                                        const BoundIter<T>& rhs)
    {
        return !(lhs > rhs);
    }

    GRACE_DEVICE friend bool operator>=(const BoundIter<T>& lhs,
                                        const BoundIter<T>& rhs)
    {
        return !(lhs < rhs);
    }
};

} // namespace gpu

} // namespace grace
