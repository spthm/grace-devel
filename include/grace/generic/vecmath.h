#pragma once

#include "grace/types.h"

#include <cmath>

namespace grace {

template <typename Real3>
GRACE_HOST_DEVICE Real3 normalize3(const Real3 v)
{
#ifdef __CUDA_ARCH__

#if __CUDACC_VER_MAJOR__ >= 7
    double N = rnorm3d(v.x, v.y, v.z);
#else
    double N = rsqrt(v.x*v.x + v.y*v.y + v.z*v.z);
#endif

#else
    double N = 1. / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
#endif

    Real3 normv;
    normv.x = v.x * N;
    normv.y = v.y * N;
    normv.z = v.z * N;

    return normv;
}

template <typename Real3>
GRACE_HOST_DEVICE double dot3(const Real3 u, const Real3 v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

// NB: only defined for three-component vectors.
template <typename Real3>
GRACE_HOST_DEVICE Real3 cross(const Real3 u, const Real3 v)
{
    Real3 result;

    result.x = u.y * v.z - u.z * v.y;
    result.y = u.z * v.x - u.x * v.z;
    result.z = u.x * v.y - u.y * v.x;

    return result;
}

} // namespace grace
