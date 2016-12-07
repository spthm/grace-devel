#pragma once

#include "grace/error.h"
#include "grace/types.h"

#include <cstddef>

namespace grace {

//
// BoundedPtr member functions
//

template <typename T>
GRACE_HOST_DEVICE
BoundedPtr<T>::BoundedPtr(char* const begin, const size_t bytes)
    : begin_(reinterpret_cast<T*>(begin)),
      end_(reinterpret_cast<T*>(begin + bytes)),
      ptr_(reinterpret_cast<T*>(begin)) {}

template <typename T>
GRACE_HOST_DEVICE
BoundedPtr<T>::BoundedPtr(char* const begin, char* const end)
    : begin_(reinterpret_cast<T*>(begin)),
      end_(reinterpret_cast<T*>(end)),
      ptr_(reinterpret_cast<T*>(begin)) {}

template <typename T, typename U>
GRACE_HOST_DEVICE
BoundedPtr<T>::BoundedPtr(const BoundedPtr<U>& other)
    : begin_(reinterpret_cast<T*>(other.begin_)),
      end_(reinterpret_cast<T*>(other.end_)),
      ptr_(reinterpret_cast<T*>(other.ptr_)) {}

template <typename T, typename U>
GRACE_HOST_DEVICE
BoundedPtr<T>& BoundedPtr<T>::operator=(const BoundedPtr<U>& other)
{
    BoundedPtr<T> other_T(other);
    swap(*this, other_T);
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
void BoundedPtr<T>::align_to(const size_t alignment)
{
    char* aligned = reinterpret_cast<char*>(ptr_);
    int rem = (uintptr_t)aligned % alignment;
    if (rem != 0) {
        aligned += (alignment - rem);
    }

    ptr_ = reinterpret_cast<T*>(aligned);
}

template <typename T>
GRACE_HOST_DEVICE
T& BoundedPtr<T>::operator*()
{
    GRACE_ASSERT(ptr_ >= begin_, boundedptr_memory_underflow);
    GRACE_ASSERT(ptr_ + sizeof(T) < end_, boundedptr_memory_overflow);

    return *ptr_;
}

template <typename T>
GRACE_HOST_DEVICE
const T& BoundedPtr<T>::operator*() const
{
    GRACE_ASSERT(ptr_ >= begin_, boundedptr_memory_underflow);
    GRACE_ASSERT(ptr_ + sizeof(T) < end_, boundedptr_memory_overflow);

    return *ptr_;
}

template <typename T>
GRACE_HOST_DEVICE
T& BoundedPtr<T>::operator[](difference_type i)
{
    return *((*this) + i);
}

template <typename T>
GRACE_HOST_DEVICE
const T& BoundedPtr<T>::operator[](difference_type i) const
{
    return *((*this) + i);
}

template <typename T>
GRACE_HOST_DEVICE
T* BoundedPtr<T>::operator->()
{
    return &(*(*this));
}

template <typename T>
GRACE_HOST_DEVICE
const T* BoundedPtr<T>::operator->() const
{
    return &(*(*this));
}

template <typename T>
GRACE_HOST_DEVICE
BoundedPtr<T>::difference_type BoundedPtr<T>::operator-(const BoundedPtr<T>& other) const
{
    return ptr_ - other.ptr_;
}

template <typename T>
GRACE_HOST_DEVICE
BoundedPtr<T>& BoundedPtr<T>::operator+=(const difference_type n)
{
    ptr_ += n;
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
BoundedPtr<T>& BoundedPtr<T>::operator-=(const difference_type n)
{
    ptr_ -= n;
    return *this;
}


//
// BoundedPtr-related friend and free functions
//

template <typename T>
GRACE_HOST_DEVICE
void swap(BoundedPtr<T>& lhs, BoundedPtr<T>& rhs)
{
    // We make unqualified calls to swap to ensure that any swap operator
    // for types T found via ADL used; we also want to use std::swap if no
    // other swap() exists.
    using std::swap;

    swap(lhs.ptr_, rhs.ptr_);
    swap(lhs.begin_, rhs.begin_);
    swap(lhs.end_, rhs.end_);
}

template <typename T>
GRACE_HOST_DEVICE
bool operator==(const BoundedPtr<T>& lhs, const BoundedPtr<T>& rhs)
{
    return lhs.ptr_ == rhs.ptr_;
}

template <typename T>
GRACE_HOST_DEVICE
bool operator<(const BoundedPtr<T>& lhs, const BoundedPtr<T>& rhs)
{
    return (rhs - lhs) > 0;
}

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
