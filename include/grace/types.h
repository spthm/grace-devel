#pragma once

#include <stdint.h>

// Final clause seems to be necessary with some versions of NVCC, where
// __CUDA_ARCH__ == 0 in host(?) compilation trajectory.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200) && (__CUDA_ARCH__ != 0)
#error GRACE does not support devices of compute capability < 2.0.
#endif

#if defined(__NVCC__) || defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define GRACE_CUDA_VERSION  __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10
#else
#define GRACE_CUDA_VERSION -1
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

// C++11 has the portable function aligned_alloc(), which should be used where
// available.
#if __cplusplus >= 201103L || defined(_ISOC11_SOURCE)
#define GRACE_USE_CPP11_ALIGNED_ALLOC
#elif _POSIX_C_SOURCE >= 200112L || (defined(__APPLE__) && defined(__MACH__))
#define GRACE_USE_POSIX_MEMALIGN
#elif defined(__INTEL_COMPILER)
#define GRACE_USE_MM_MALLOC
#elif defined (MSC_VER)
#define GRACE_USE_MS_ALIGNED_MALLOC
#else
#error Compiler or system not detected, no aligned malloc available.
#endif

// C++11 has the portable syntax
//   alignof(T)
// which should preferentially be used where available.
#if __cplusplus >= 201103L
#define GRACE_ALIGNOF(T) alignof(T)
#elif defined(__CUDACC__)
#define GRACE_ALIGNOF(T) __alignof(T)
#elif defined(__GNUC__)
#define GRACE_ALIGNOF(T) __alignof__ (T)
#elif defined(__clang__)
#define GRACE_ALIGNOF(T) __alignof__(T)
#elif defined(__INTEL_COMPILER)
#define GRACE_ALIGNOF(T) __alignof(T)
#elif defined(_MSC_VER)
#define GRACE_ALIGNOF(T) __alignof(T)
#else
#error Compiler not detected, no alignof function available.
#endif

// C++11 has the portable syntax
//   struct alignas(16) foo { ... }
// which should preferentially be used where available.
// NVCC, where this is most important (to enable efficient 2- and 4-wide loads
// on the device) as the simple __align__(16) syntax,
//   http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
// GCC (and clang?) have __attribute__
//   https://gcc.gnu.org/onlinedocs/gcc/Type-Attributes.html#Type-Attributes
// MSVC has __declspec
//   https://msdn.microsoft.com/en-us/library/83ythb65.aspx
// And Intel supports both GCC and MSVC formats.
#if __cplusplus >= 201103L
#define GRACE_ALIGNAS(x) alignas(x)
#elif defined(__CUDACC__)
#define GRACE_ALIGNAS(x) __align__(x)
#elif defined(__GNUC__) || defined (__clang__) || defined(__INTEL_COMPILER)
#define GRACE_ALIGNAS(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#define GRACE_ALIGNAS(x) __declspec(align(x))
#else
#warning Compiler not detected, struct alignment may be inconsistent.
#define GRACE_ALIGNAS(x)
#endif

namespace grace {

#if defined(__GNUC__) || defined(__clang__) || defined(__APPLE__) && defined(__MACH__)
typedef uint32_t uinteger32;
typedef uint64_t uinteger64;
typedef int32_t integer32;
typedef int64_t integer64;
#elif defined(_MSC_VER)
typedef unsigned __int32 uinteger32;
typedef unsigned __int64 uinteger64;
typedef __int32 integer32;
typedef __int64 integer64;
#else
#warning Compiler or system not detected, key bit-lengths may be incorrect
typedef unsigned int uinteger32;
typedef unsigned long long uinteger64;
typedef int integer32;
typedef long long integer64;
#endif

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

// ParallelRays -> Multiple rays are tested, in parallel, against a single
//                 primitive (default).
// ParallelPrimitives -> Multiple primitives are tested, in parallel, against a
//                       single ray. This has some overhead.
struct LeafTraversal {
    enum E {
        ParallelRays,
        ParallelPrimitives
    };
};

} // namespace grace

#include "grace/detail/math_shim.h"
