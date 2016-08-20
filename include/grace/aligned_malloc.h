#pragma once

#include <stdlib.h>

#if defined(__clang__)
#include <mm_malloc.h>
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)
#include <malloc.h>
#endif

namespace grace {

template <typename T>
T* aligned_malloc(const size_t size,
                  const size_t alignment = GRACE_ALIGNOF(T))
{
    T* memptr;

#if __cplusplus >= 201103L
    memptr = (T*)aligned_alloc(alignment, size);

#elif defined(__GNUC__)
    posix_memalign((void**)&memptr, alignment, size);

#elif defined (__INTEL_COMPILER) || defined(__clang__)
    memptr = (T*)_mm_malloc(size, alignment);

#elif defined(_MSC_VER)
    memptr = (T*)_aligned_malloc(size, alignment);

#else
    #error Compiler not detected, no aligned_malloc function available.

#endif

    return memptr;
}

template <typename T>
void aligned_free(T* memptr)
{
#if defined(__INTEL_COMPILER) || defined(__clang__)
    _mm_free((void*)memptr);

#else
    free(memptr);

#endif
}

} // namespace grace
