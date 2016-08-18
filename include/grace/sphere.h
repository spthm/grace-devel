#pragma once

#include "grace/types.h"

namespace grace {

template <typename T>
GRACE_ALIGNED_STRUCT(16) Sphere
{
    typedef T base_type;

    T x, y, z, r;

    GRACE_HOST_DEVICE Sphere();

    GRACE_HOST_DEVICE Sphere(T x, T y, T z, T r);

#ifdef __CUDACC__
    GRACE_HOST_DEVICE Sphere(float4 xyzw);

    GRACE_HOST_DEVICE Sphere(double4 xyzw);
#endif
};

} //namespace grace

#include "grace/detail/sphere-inl.h"
