#pragma once

#include "grace/types.h"
#include "grace/aligned_malloc.h"

#include <stdlib.h>

#if defined(GRACE_USE_MM_MALLOC) || defined(GRACE_USE_MS_ALIGNED_MALLOC)
#include <malloc.h>
#endif

namespace grace {

GRACE_HOST void* aligned_malloc(const size_t size, const size_t alignment)
{
    void* memptr = NULL;

#if defined(GRACE_USE_CPP11_ALIGNED_ALLOC)
    memptr = aligned_alloc(alignment, size);

#elif defined(GRACE_USE_POSIX_MEMALIGN)
    int res;
    res = posix_memalign(&memptr, alignment, size);
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
