#pragma once

#include "grace/generic/meta.h"

#include "grace/error.h"
#include "grace/types.h"

#include <cstddef>
#include <iterator>

namespace grace {

// A pointer which may only access the underlying array in a specific range.
// For efficiency, range checking only occurs in debug mode.
// No checks are made on the alignment of the pointer; in order to ensure
// correct alignment, multiple BoundedPtrs should be declared in order of
// decreasing alignment requirements, just as one would do for a raw pointer.
//
// For example, if we wish to store 5 doubles, 6 floats and 2 chars, then,
// supposing char* ptr points to memory of sufficient size and satisfies the
// alignment requirements of double,
//     const size_t size = 5*sizeof(double) + 6*sizeof(float) + 2*sizeof(char)
//     bptr_d = BoundedPtr<double>(ptr, size);
//     bptr_f = BoundedPtr<float>(bptr_d + 5);
//     bptr_c = BoundedPtr<char>(bptr_f + 6);
// ensures correct alignment for all bptr_Xs, and *(bptr_c + 1) is within the
// bounds implied by ptr and size.
//
// BoundedPtr has an align_to() method to set the alignment, but note that this
// will either not modify the underlying pointer, or it will increment it, and
// hence may result in unexpected out-of-bounds accesses.
//
// Alignment requirements of a type can be checked with the GRACE_ALIGNOF macro.
template <typename T>
class BoundedPtr
{
public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef T value_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;

    template <typename U>
    friend class BoundedPtr;

    GRACE_HOST_DEVICE BoundedPtr(char* const begin, const size_t bytes)
        : begin_(reinterpret_cast<T*>(begin)),
          end_(reinterpret_cast<T*>(begin + bytes)),
          ptr_(reinterpret_cast<T*>(begin)) {}

    // Note this is not the copy constructor, because a copy constructor cannot
    // be a template. We use the default copy constructor. It is explicit, so it
    // is also not a converting constructor; explicit because e.g. we don't want
    // operator+() to work for BoundedPtrs with different value_types.
    template <typename U>
    GRACE_HOST_DEVICE explicit BoundedPtr(const BoundedPtr<U>& other)
        : begin_(reinterpret_cast<T*>(other.begin_)),
          end_(reinterpret_cast<T*>(other.end_)),
          ptr_(reinterpret_cast<T*>(other.ptr_)) {}

    // Again, copy-assigment operator must not be a templte. We use the default
    // copy-assignment operator.
    template <typename U>
    GRACE_HOST_DEVICE BoundedPtr<T>& operator=(const BoundedPtr<U>& other)
    {
        BoundedPtr<T> other_T(other);
        swap(*this, other_T);
        return *this;
    }

    // Ensures alignment, increasing address when necessary.
    GRACE_HOST_DEVICE void align_to(const size_t alignment)
    {
        char* aligned = reinterpret_cast<char*>(ptr_);
        int rem = (uintptr_t)aligned % alignment;
        if (rem != 0) {
            aligned += (alignment - rem);
        }

        ptr_ = reinterpret_cast<T*>(aligned);
    }

    // It doesn't make sense to swap BoundedPtrs of different value_types.
    friend void swap(BoundedPtr<T>& lhs, BoundedPtr<T>& rhs)
    {
        // We make unqualified calls to swap to ensure that any swap operator
        // for types T found via ADL used; we also want to use std::swap if no
        // other swap() exists.
        using std::swap;

        swap(lhs.ptr_, rhs.ptr_);
        swap(lhs.begin_, rhs.begin_);
        swap(lhs.end_, rhs.end_);
    }

    GRACE_HOST_DEVICE T& operator*()
    {
        GRACE_ASSERT(ptr_ >= begin_, boundedptr_memory_underflow);
        GRACE_ASSERT(ptr_ + sizeof(T) < end_, boundedptr_memory_overflow);

        return *ptr_;
    }

    GRACE_HOST_DEVICE const T& operator*() const
    {
        GRACE_ASSERT(ptr_ >= begin_, boundedptr_memory_underflow);
        GRACE_ASSERT(ptr_ + sizeof(T) < end_, boundedptr_memory_overflow);

        return *ptr_;
    }

    GRACE_HOST_DEVICE T& operator[](difference_type i)
    {
        return *((*this) + i);
    }

    GRACE_HOST_DEVICE const T& operator[](difference_type i) const
    {
        return *((*this) + i);
    }

    GRACE_HOST_DEVICE T* operator->()
    {
        return &(*(*this));
    }

    GRACE_HOST_DEVICE const T* operator->() const
    {
        return &(*(*this));
    }

    GRACE_HOST_DEVICE difference_type operator-(const BoundedPtr<T>& other) const
    {
        return ptr_ - other.ptr_;
    }

    GRACE_HOST_DEVICE BoundedPtr<T>& operator+=(const difference_type n)
    {
        ptr_ += n;
        return *this;
    }

    GRACE_HOST_DEVICE BoundedPtr<T>& operator-=(const difference_type n)
    {
        ptr_ -= n;
        return *this;
    }

    GRACE_HOST_DEVICE friend bool operator<(const BoundedPtr<T>& lhs,
                                            const BoundedPtr<T>& rhs)
    {
        return (rhs - lhs) > 0;
    }

    // Could be implemented from above; not done so for efficiency.
    GRACE_HOST_DEVICE friend bool operator==(const BoundedPtr<T>& lhs,
                                             const BoundedPtr<T>& rhs)
    {
        return lhs.ptr_ == rhs.ptr_;
    }

private:
    // Same type so they may safely be compared.
    // begin_ and end_ correspond to the total bounded memory range, and hence
    // are not necessarily aligned for a type T, and should _never_ be
    // dereferenced.
    T* begin_;
    T* end_;
    T* ptr_;
};

// Prefix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T>& operator++(BoundedPtr<T>& bptr)
{
    bptr += 1;
    return bptr;
}

// Postfix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator++(BoundedPtr<T>& bptr, int)
{
    BoundedPtr<T> prev = bptr;
    ++bptr;
    return prev;
}

// Prefix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T>& operator--(BoundedPtr<T>& bptr)
{
    bptr -= 1;
    return bptr;
}

// Postfix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator--(BoundedPtr<T>& bptr, int)
{
   BoundedPtr<T> prev = bptr;
   --bptr;
   return prev;
}

GRACE_HOST_DEVICE BoundedPtr<T> operator+(const BoundedPtr<T>& lhs,
                                          const difference_type rhs)
{
    BoundedPtr<T> bptr = lhs;
    bptr += rhs;
    return bptr;
}

GRACE_HOST_DEVICE BoundedPtr<T> operator+(const difference_type lhs,
                                          const BoundedPtr<T>& rhs)
{
    // Swap sides.
    return rhs + lhs;
}

GRACE_HOST_DEVICE BoundedPtr<T> operator-(const BoundedPtr<T>& lhs,
                                          const difference_type rhs)
{
    return lhs + (-rhs);
}

GRACE_HOST_DEVICE bool operator!=(const BoundedPtr<T>& lhs,
                                  const BoundedPtr<T>& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator>(const BoundedPtr<T>& lhs,
                                 const BoundedPtr<T>& rhs)
{
    // Swap sides.
    return rhs < lhs;
}

GRACE_HOST_DEVICE bool operator<=(const BoundedPtr<T>& lhs,
                                  const BoundedPtr<T>& rhs)
{
    return !(lhs > rhs);
}

GRACE_HOST_DEVICE bool operator>=(const BoundedPtr<T>& lhs,
                                  const BoundedPtr<T>& rhs)
{
    return !(lhs < rhs);
}

} // namespace grace
