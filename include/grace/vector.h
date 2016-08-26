#pragma once

#include "grace/sphere.h"
#include "grace/types.h"

#include "grace/detail/vector_members-inl.h"

namespace grace {

// Circular dependency.
template <typename T>
struct Sphere;

// Defining a vector with Dims = {2, 3, 4} with sizeof(T) >= 9 (excepting
// sizeof(T) == 16) results in .data() and operator[] implementations which are
// not reliable. Specifically, *(vec.data() + i) and vec[i] are guaranteed to
// return the correct value if and only if i == 0.
template <size_t Dims, typename T>
struct Vector;

// template <typename T>
// GRACE_ALIGNED_STRUCT(16) Vector<3, T>
// {
//     typedef T value_type;

//     T x;
//     T y;
//     T z;
template <typename T>
struct Vector<3, T> : detail::VectorMembers<3, sizeof(T), T>
{
    typedef T value_type;

    // Zero-initialized.
    GRACE_HOST_DEVICE Vector() :
        Base() {}

    GRACE_HOST_DEVICE Vector(const T x, const T y, const T z) :
        Base(x, y, z) {}

    GRACE_HOST_DEVICE Vector(const T s) :
        Base(s, s, s) {}

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const U data[3]) :
        Base(data[0], data[1], data[2]) {}

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<3, U>& vec) :
        Base(vec.x, vec.y, vec.z) {}

    // vec.w == vec[3] ignored.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<4, U>& vec) :
        Base(vec.x, vec.y, vec.z) {}

    // sphere.r ignored.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Sphere<U>& sphere) :
        Base(sphere.x, sphere.y, sphere.z) {}

#ifdef __CUDACC__
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float3& xyz) :
        Base(xyz.x, xyz.y, xyz.z) {}

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double3& xyz) :
        Base(xyz.x, xyz.y, xyz.z) {}

    // w ignored.
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float4& xyzw) :
        Base(xyzw.z, xyzw.y, xyzw.z) {}

    // w ignored.
    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double4& xyzw) :
        Base(xyzw.z, xyzw.y, xyzw.z) {}
#endif

    template <typename U>
    GRACE_HOST_DEVICE Vector& operator=(const Vector<3, U>& rhs);

    // Generally unsafe, but detail::VectorMember<3, T> ensures it is safe iff
    // T is a 1, 2, 4 or 8-byte type.
    GRACE_HOST_DEVICE T* data();
    GRACE_HOST_DEVICE const T* data() const;

    // Generally unsafe, but detail::VectorMember<3, T> ensures it is safe iff
    // T is a 1, 2, 4 or 8-byte type.
    GRACE_HOST_DEVICE T& operator[](size_t i);
    GRACE_HOST_DEVICE const T& operator[](size_t i) const;

private:
    typedef detail::VectorMembers<3, sizeof(T), T> Base;
};

// template <typename T>
// GRACE_ALIGNED_STRUCT(16) Vector<4, T>
// {
//     typedef T value_type;

//     T x;
//     T y;
//     T z;
//     T w;
template <typename T>
struct Vector<4, T> : detail::VectorMembers<4, sizeof(T), T>
{
    typedef T value_type;

    // Zero-initialized.
    GRACE_HOST_DEVICE Vector() :
        Base() {}

    GRACE_HOST_DEVICE Vector(const T x, const T y, const T z, const T w) :
        Base(x, y, z, w) {}

    GRACE_HOST_DEVICE Vector(const T s) :
        Base(s, s, s, s) {}

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const U data[4]) :
        Base(data[0], data[1], data[2], data[3]) {}

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<4, U>& vec) :
        Base(vec.x, vec.y, vec.z, vec.w) {}

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<3, U>& vec, const U w) :
        Base(vec.x, vec.y, vec.z, w) {}

    // sphere.r is vec.w == vec[3].
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Sphere<U>& sphere) :
        Base(sphere.x, sphere.y, sphere.z, sphere.w) {}

#ifdef __CUDACC__
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float3& xyz, const float w) :
        Base(xyz.x, xyz.y, xyz.z, w) {}

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double3& xyz, const double w) :
        Base(xyz.x, xyz.y, xyz.z, w) {}

    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float4& xyzw) :
        Base(xyzw.x, xyzw.y, xyzw.z, xyzw.w) {}

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double4& xyzw) :
        Base(xyzw.x, xyzw.y, xyzw.z, xyzw.w) {}
#endif

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector& operator=(const Vector<4, U>& rhs);

    // Generally unsafe, but detail::VectorMember<3, T> ensures it is safe iff
    // T is a 1, 2, 4 or 8-byte type.
    GRACE_HOST_DEVICE T* data();
    GRACE_HOST_DEVICE const T* data() const;

    // Generally unsafe, but detail::VectorMember<3, T> ensures it is safe iff
    // T is a 1, 2, 4 or 8-byte type.
    GRACE_HOST_DEVICE T& operator[](size_t i);
    GRACE_HOST_DEVICE const T& operator[](size_t i) const;

    GRACE_HOST_DEVICE Vector<3, T>& vec3();
    GRACE_HOST_DEVICE const Vector<3, T>& vec3() const;

private:
    typedef detail::VectorMembers<4, sizeof(T), T> Base;
};


/* Vector operations */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& vec);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T max_element(const Vector<Dims, T>& vec);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T min_element(const Vector<Dims, T>& vec);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T norm2(const Vector<Dims, T>& vec);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T norm(const Vector<Dims, T>& vec);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> normalize(const Vector<Dims, T>& vec);


/* Vector-vector operations */

// Cross-product only definied for length-three vectors.
template <typename T>
GRACE_HOST_DEVICE
Vector<3, T> cross(const Vector<3, T>& u, const Vector<3, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> dot(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> max(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> min(const Vector<Dims, T>& u, const Vector<Dims, T>& v);


/* Vector-vector arithmetic */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const Vector<Dims, T>& u, const Vector<Dims, T>& v);


/* Vector-scalar arithmetic */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const Vector<Dims, T>& v, const T s);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& v, const T s);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const Vector<Dims, T>& v, const T s);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const Vector<Dims, T>& v, const T s);


/* Scalar-Vector arithmetic */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const T s, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const T s, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const T s, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const T s, const Vector<Dims, T>& v);

} //namespace grace

#include "grace/detail/vectormath-inl.h"
#include "grace/detail/vector3-inl.h"
#include "grace/detail/vector4-inl.h"
