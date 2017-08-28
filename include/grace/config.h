#pragma once

// Required for the various _POSIX_C_SOURCE etc. macros.
#include <cstdlib>

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

// Don't include __NVCC__ here, becuase that's defined even if nvcc is
// compiling a C/C++ file. We quite specifically want to mark CUDA as available
// only when compiling cuda source files for device- and host-trajectory.
// __CUDA__ is a clang macro.
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA__)
#define GRACE_CUDA_AVAILABLE
#endif

#if defined(GRACE_CUDA_AVAILABLE)
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

// C++17 has the portable function aligned_alloc(), which should be used where
// available.
#if __cplusplus >= 201703L
#define GRACE_USE_CPP17_ALIGNED_ALLOC
#elif __STDC_VERSION__ >= 201112L
#define GRACE_USE_C11_ALIGNED_ALLOC
#elif _POSIX_C_SOURCE >= 200112L || (defined(__APPLE__) && defined(__MACH__))
#define GRACE_USE_POSIX_MEMALIGN
#elif defined(__INTEL_COMPILER)
#define GRACE_USE_MM_MALLOC
#elif defined (_MSC_VER)
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

// Want this to be included everywhere.
#include "grace/detail/math_shim.h"
