#pragma once

#include "grace/vector.h"
#include "grace/types.h"

namespace grace {

// Circular dependency.
template <size_t Dims, typename T>
struct Vector;

template <typename T>
GRACE_ALIGNED_STRUCT(16) Sphere
{
    typedef T value_type;

    T x;
    T y;
    T z;
    T r;

    // Unit sphere at origin.
    GRACE_HOST_DEVICE Sphere();

    GRACE_HOST_DEVICE Sphere(const T x, const T y, const T z, const T r);

    GRACE_HOST_DEVICE Sphere(const T s);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Sphere(const U data[4]);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Sphere(const Sphere<U>&);

    // Unit sphere.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Sphere(const Vector<3, U>& centre);

    // U must be convertible to T.
    template <typename U, typename S>
    GRACE_HOST_DEVICE Sphere(const Vector<3, U>& centre, S radius);

    // vec.w == vec[3] is radius.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Sphere(const Vector<4, U>& vec);

#ifdef __CUDACC__
    // Unit sphere.
    // float must be convertible to T.
    GRACE_HOST_DEVICE Sphere(const float3& xyz);

    // Unit sphere.
    // double must be convertible to T.
    GRACE_HOST_DEVICE Sphere(const double3& xyz);

    // w is radius.
    // float must be convertible to T.
    GRACE_HOST_DEVICE Sphere(const float4& xyzw);

    // w is radius.
    // double must be convertible to T.
    GRACE_HOST_DEVICE Sphere(const double4& xyzw);
#endif
};

} //namespace grace

#include "grace/detail/sphere-inl.h"