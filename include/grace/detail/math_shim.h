#pragma once

#include <algorithm>

namespace grace {

/* grace defines grace::min() and grace::max() in detail/vector_math-inl.h.
 * These template functions are picked up by the compiler whenever min() or
 * max() are called from within the grace:: namespace, but of course fail (at
 * compile time) for all types which are not grace::Vector<N, T>.
 * std::min() and std::max() are the appropriate functions to call in host-only
 * code, and do not suffer from this problem. However, neither can be called
 * from device code, thus all __host__ __device__ functions are also affected.
 *
 * As a fix, every call to min() and max() could be replaced with calls to
 * ::min() and ::max(), but this is obviously error prone. The below
 * grace::min() and grace::max() functions serve as wrappers for this purpose;
 * they will be seen by the compiler when in the grace:: namespace, and delegate
 * the appropriate function call.
 * All overloads of min() and max() in the include/math_functions.h header of
 * CUDA-7.5 are repeated here. Calls to grace::min() and grace::max() from
 * __device__ code thus resolve to their CUDA-specific builtins.
 * For consistency, calls from the host will resolve instead to std::min() and
 * std::max(), which are still available when CUDA is not.
 *
 * Note that nvcc includes cuda_runtime.h by default, which includes
 * common_functions.h, which includes the aforementioned math_functions.h.
 * (See e.g. http://stackoverflow.com/a/29710576.)
 * Since we only required overloaded ::min() and ::max() when following the
 * device compilation trajectory, we can be sure they exist without #includ-ing
 * any other headers.
 *
 * Finally, note that none of this works if all this code is in namespace
 * grace::detail:: - it _MUST_ be in namespace grace::
 */


//
// min()
//

GRACE_HOST_DEVICE int min(int a, int b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned int min(unsigned int a, unsigned int b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned int min(unsigned int a, int b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, (unsigned int)b);
#else
    return std::min(a, (unsigned int)b);
#endif
}

GRACE_HOST_DEVICE unsigned int min(int a, unsigned int b)
{
#ifdef __CUDA_ARCH__
    return ::min((unsigned int)a, b);
#else
    return std::min((unsigned int)a, b);
#endif
}

GRACE_HOST_DEVICE long min(long a, long b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long min(unsigned long a, unsigned long b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long min(unsigned long a, long b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, (unsigned long)b);
#else
    return std::min(a, (unsigned long)b);
#endif
}

GRACE_HOST_DEVICE unsigned long min(long a, unsigned long b)
{
#ifdef __CUDA_ARCH__
    return ::min((unsigned long)a, b);
#else
    return std::min((unsigned long)a, b);
#endif
}

GRACE_HOST_DEVICE long long min(long long a, long long b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long long min(unsigned long long a, unsigned long long b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long long min(unsigned long long a, long long b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, (unsigned long long)b);
#else
    return std::min(a, (unsigned long long)b);
#endif
}

GRACE_HOST_DEVICE unsigned long long min(long long a, unsigned long long b)
{
#ifdef __CUDA_ARCH__
    return ::min((unsigned long long)a, b);
#else
    return std::min((unsigned long long)a, b);
#endif
}

GRACE_HOST_DEVICE float min(float a, float b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE double min(double a, double b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
}

GRACE_HOST_DEVICE double min(double a, float b)
{
#ifdef __CUDA_ARCH__
    return ::min(a, (double)b);
#else
    return std::min(a, (double)b);
#endif
}

GRACE_HOST_DEVICE double min(float a, double b)
{
#ifdef __CUDA_ARCH__
    return ::min((double)a, b);
#else
    return std::min((double)a, b);
#endif
}


//
// max()
//

GRACE_HOST_DEVICE int max(int a, int b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned int max(unsigned int a, unsigned int b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned int max(unsigned int a, int b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, (unsigned int)b);
#else
    return std::max(a, (unsigned int)b);
#endif
}

GRACE_HOST_DEVICE unsigned int max(int a, unsigned int b)
{
#ifdef __CUDA_ARCH__
    return ::max((unsigned int)a, b);
#else
    return std::max((unsigned int)a, b);
#endif
}

GRACE_HOST_DEVICE long max(long a, long b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long max(unsigned long a, unsigned long b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long max(unsigned long a, long b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, (unsigned long)b);
#else
    return std::max(a, (unsigned long)b);
#endif
}

GRACE_HOST_DEVICE unsigned long max(long a, unsigned long b)
{
#ifdef __CUDA_ARCH__
    return ::max((unsigned long)a, b);
#else
    return std::max((unsigned long)a, b);
#endif
}

GRACE_HOST_DEVICE long long max(long long a, long long b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long long max(unsigned long long a, unsigned long long b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE unsigned long long max(unsigned long long a, long long b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, (unsigned long long)b);
#else
    return std::max(a, (unsigned long long)b);
#endif
}

GRACE_HOST_DEVICE unsigned long long max(long long a, unsigned long long b)
{
#ifdef __CUDA_ARCH__
    return ::max((unsigned long long)a, b);
#else
    return std::max((unsigned long long)a, b);
#endif
}

GRACE_HOST_DEVICE float max(float a, float b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE double max(double a, double b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
}

GRACE_HOST_DEVICE double max(double a, float b)
{
#ifdef __CUDA_ARCH__
    return ::max(a, (double)b);
#else
    return std::max(a, (double)b);
#endif
}

GRACE_HOST_DEVICE double max(float a, double b)
{
#ifdef __CUDA_ARCH__
    return ::max((double)a, b);
#else
    return std::max((double)a, b);
#endif
}


} // namespace grace
