#pragma once

#include <stdint.h>

// CUDA
#include <vector_types.h>

// Final clause seems to be necessary with some versions of NVCC, where
// __CUDA_ARCH__ == 0 in host(?) compilation trajectory.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200) && (__CUDA_ARCH__ != 0)
#error GRACE does not support devices of compute capability < 2.0.
#endif

#ifdef __CUDACC__
    #define GRACE_HOST __host__ inline
    #define GRACE_DEVICE __device__ inline
    #define GRACE_HOST_DEVICE __host__ __device__ inline
#else
    // GRACE_DEVICE does not make sense as an identifier when compiling non-CUDA
    // source files. It should (and will) cause the compiler to balk.
    // GRACE_HOST[_DEVICE], however, should compile just fine.
    #define GRACE_HOST inline
    #define GRACE_DEVICE __device__ inline
    #define GRACE_HOST_DEVICE inline
#endif

namespace grace {

typedef uint32_t uinteger32;
typedef uint64_t uinteger64;
typedef int32_t integer32;
typedef int64_t integer64;

// Binary encoding with +ve = 1, -ve = 0.
// Octants for ray generation.
enum Octants {
    PPP = 7,
    PPM = 6,
    PMP = 5,
    PMM = 4,
    MPP = 3,
    MPM = 2,
    MMP = 1,
    MMM = 0
};

enum RaySortType {
    NoSort,
    DirectionSort,
    EndPointSort
};

} // namespace grace
