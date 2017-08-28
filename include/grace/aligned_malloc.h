#pragma once

#include "grace/config.h"

namespace grace {

// Allocates size bytes of memory alligned to alignment bytes.
GRACE_HOST void* aligned_malloc(const size_t size, const size_t alignment);

// Memory allocated with aligned_malloc must be free'd with aligned_free.
GRACE_HOST void aligned_free(void* memptr);

} // namespace grace

#include "grace/detail/aligned_malloc-inl.h"
