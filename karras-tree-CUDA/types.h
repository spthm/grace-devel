#pragma once

#include <stdint.h>

#ifdef __CUDACC__
    #define GRACE_HOST __host__ inline
    #define GRACE_DEVICE __device__ inline
    #define GRACE_HOST_DEVICE __host__ __device__ inline
#endif

namespace grace {

typedef uint32_t uinteger32;
typedef uint64_t uinteger64;
typedef int32_t integer32;
typedef int64_t integer64;

} // namespace grace
