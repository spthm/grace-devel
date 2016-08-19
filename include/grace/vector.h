#pragma once

#include "grace/sphere.h"
#include "grace/types.h"

namespace grace {

// Circular dependency.
template <typename T>
struct Sphere;

template <size_t Dims, typename T>
struct Vector;

template <typename T>
GRACE_ALIGNED_STRUCT(16) Vector<3, T>
{
    typedef T value_type;

    T x;
    T y;
    T z;

    // Zero-initialized.
    GRACE_HOST_DEVICE Vector();

    GRACE_HOST_DEVICE Vector(const T x, const T y, const T z);

    GRACE_HOST_DEVICE Vector(const T s);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const U data[3]);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<3, U>& vec);

    // vec.w == vec[3] ignored.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<4, U>& vec);

    // sphere.r ignored.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Sphere<U>& sphere);

#ifdef __CUDACC__
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float3& xyz);

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double3& xyz);

    // w ignored.
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float4& xyzw);

    // w ignored.
    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double4& xyzw);
#endif

    template <typename U>
    GRACE_HOST_DEVICE Vector& operator=(const Vector<3, U>& rhs);

    GRACE_HOST_DEVICE T* get();
    GRACE_HOST_DEVICE const T* get() const;

    GRACE_HOST_DEVICE T& operator[](size_t i);
    GRACE_HOST_DEVICE const T& operator[](size_t i) const;
};

template <typename T>
GRACE_ALIGNED_STRUCT(16) Vector<4, T>
{
    typedef T value_type;

    T x;
    T y;
    T z;
    T w;

    // Zero-initialized.
    GRACE_HOST_DEVICE Vector();

    GRACE_HOST_DEVICE Vector(const T x, const T y, const T z, const T w);

    GRACE_HOST_DEVICE Vector(const T s);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const U data[4]);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<4, U>& vec);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<3, U>& vec, const U w);

    // sphere.r is vec.w == vec[3].
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Sphere<U>& sphere);

#ifdef __CUDACC__
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float3& xyz, const float w);

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double3& xyz, const double w);

    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float4& xyzw);

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double4& xyzw);
#endif

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector& operator=(const Vector<4, U>& rhs);

    GRACE_HOST_DEVICE T* get();
    GRACE_HOST_DEVICE const T* get() const;

    GRACE_HOST_DEVICE T& operator[](size_t i);
    GRACE_HOST_DEVICE const T& operator[](size_t i) const;

    GRACE_HOST_DEVICE Vector<3, T>& vec3();
    GRACE_HOST_DEVICE const Vector<3, T>& vec3() const;
};

} //namespace grace

#include "grace/detail/vector3-inl.h"
#include "grace/detail/vector4-inl.h"
