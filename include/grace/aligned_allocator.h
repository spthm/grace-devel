#pragma once

#include "grace/aligned_malloc.h"
#include "grace/config.h"

#include <cstddef>
#include <memory>

namespace grace {

template <typename T, size_t alignment>
struct aligned_allocator
{
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, alignment> other;
    };

    GRACE_HOST aligned_allocator() {}
    GRACE_HOST aligned_allocator(const aligned_allocator& other) {}

    template <typename U>
    GRACE_HOST aligned_allocator(const aligned_allocator<U, alignment>& other) {}

    GRACE_HOST ~aligned_allocator() {}

    GRACE_HOST pointer address(reference x) const;
    GRACE_HOST const_pointer address(const_reference x) const;

    // Can throw std::bad_alloc.
    // hint is unused.
    GRACE_HOST pointer allocate(
        size_type n,
        typename std::allocator<void>::const_pointer hint = 0);


    GRACE_HOST void deallocate(pointer p, size_type n);

    GRACE_HOST size_type max_size() const;

    GRACE_HOST void construct(pointer p, const_reference val);

    GRACE_HOST void destroy(pointer p);
};

// If two allocators are equal, one can deallocate memory allocated with the
// other.
template <typename T1, size_t alignment1, typename T2, size_t alignment2>
GRACE_HOST bool operator==(const aligned_allocator<T1, alignment1>&,
                           const aligned_allocator<T2, alignment2>&);

template <typename T1, size_t alignment1, typename T2, size_t alignment2>
GRACE_HOST bool operator!=(const aligned_allocator<T1, alignment1>&,
                           const aligned_allocator<T2, alignment2>&);

} // namespace grace

#include "grace/detail/aligned_allocator-inl.h"
