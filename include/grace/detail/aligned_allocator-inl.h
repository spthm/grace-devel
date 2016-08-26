#pragma once

#include "grace/aligned_malloc.h"
#include "grace/types.h"

#include <limits>
#include <memory>
#include <new>

namespace grace {

template <typename T, size_t alignment>
GRACE_HOST typename aligned_allocator<T, alignment>::pointer
aligned_allocator<T, alignment>::address(reference x) const
{
    return &x;
}

template <typename T, size_t alignment>
GRACE_HOST typename aligned_allocator<T, alignment>::const_pointer
aligned_allocator<T, alignment>::address(const_reference x) const
{
    return &x;
}

template <typename T, size_t alignment>
GRACE_HOST typename aligned_allocator<T, alignment>::pointer
aligned_allocator<T, alignment>::allocate(
    size_type n,
    typename std::allocator<void>::const_pointer hint)
{
    pointer memptr
        = static_cast<pointer>(aligned_malloc(n * sizeof(T), alignment));

    if(memptr == NULL)
        throw std::bad_alloc();

    return memptr;
}

template <typename T, size_t alignment>
GRACE_HOST void
aligned_allocator<T, alignment>::deallocate(pointer p, size_type)
{
    // aligned_free does nothing if p == NULL.
    aligned_free(p);
}

template <typename T, size_t alignment>
GRACE_HOST typename aligned_allocator<T, alignment>::size_type
aligned_allocator<T, alignment>::max_size() const
{
    std::numeric_limits<size_type>::max() / sizeof(value_type);
}

template <typename T, size_t alignment>
GRACE_HOST void
aligned_allocator<T, alignment>::construct(pointer p, const_reference val)
{
    new((void*)p) value_type(val);
}

template <typename T, size_t alignment>
GRACE_HOST void
aligned_allocator<T, alignment>::destroy(pointer p)
{
    p->~value_type();
}

// If two allocators are equal, one can deallocate memory allocated with the
// other. Hence any two aligned_alloctor are always equal.
template <typename T1, size_t alignment1, typename T2, size_t alignment2>
GRACE_HOST bool operator==(const aligned_allocator<T1, alignment1>&,
                           const aligned_allocator<T2, alignment2>&)
{
    return true;
}

template <typename T1, size_t alignment1, typename T2, size_t alignment2>
GRACE_HOST bool operator!=(const aligned_allocator<T1, alignment1>& lhs,
                           const aligned_allocator<T2, alignment2>& rhs)
{
    return !(lhs == rhs);
}

} // namespace grace
