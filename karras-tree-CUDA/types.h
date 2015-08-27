#pragma once

#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error GRACE does not support devices of compute capability < 2.0.
#endif

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
