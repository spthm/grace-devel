#pragma once

// No grace/aligned_malloc.h include.
// This should only ever be included by grace/aligned_malloc.h.

#if defined(GRACE_USE_CPP17_ALIGNED_ALLOC)
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#if defined(GRACE_USE_MM_MALLOC) || defined(GRACE_USE_MS_ALIGNED_MALLOC)
#include <malloc.h>
#endif

namespace grace {

GRACE_HOST void* aligned_malloc(const size_t size, const size_t alignment)
{
    // Require non-zero, power-of-two.
    if ( alignment == 0 || ((alignment & (alignment - 1)) != 0) )
        return NULL;

    void* memptr = NULL;

#if defined(GRACE_USE_CPP17_ALIGNED_ALLOC)
    memptr = std::aligned_alloc(alignment, size);

#elif defined(GRACE_USE_C11_ALIGNED_ALLOC)
    memptr = aligned_alloc(alignment, size);

#elif defined(GRACE_USE_POSIX_MEMALIGN)
    // posix_memalign fails if alignment is not a multiple of sizeof(void*).
    // Where alignment is less than this value, we may safely use
    // sizeof(void*) alignment because alignments are always powers of two.
    const size_t posix_alignment
        = sizeof(void*) > alignment ? sizeof(void*) : alignment;

    int res;
    res = posix_memalign(&memptr, posix_alignment, size);
    if (res != 0) {
        memptr = NULL;
    }

#elif defined (GRACE_USE_MM_MALLOC)
    memptr = _mm_malloc(size, alignment);

#elif defined(GRACE_USE_MS_ALIGNED_MALLOC)
    memptr = _aligned_malloc(size, alignment);

#else
    #error Compiler or system not detected, no aligned malloc available.

#endif

    return memptr;
}

GRACE_HOST void aligned_free(void* memptr)
{
#if defined(GRACE_USE_MM_MALLOC)
    _mm_free(memptr);

#else
    free(memptr);

#endif
}

} // namespace grace
