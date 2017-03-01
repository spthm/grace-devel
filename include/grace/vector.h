#pragma once

#include "grace/types.h"

#include "grace/detail/vector_base-inl.h"

namespace grace {

template <size_t Dims, typename T>
struct Vector;

template <typename T>
struct Vector<3, T> : detail::vector_base<3, sizeof(T), T>
{
    typedef T value_type;

    T x;
    T y;
    T z;

    // Zero-initialized.
    GRACE_HOST_DEVICE Vector();

    GRACE_HOST_DEVICE Vector(const T x, const T y, const T z);

    GRACE_HOST_DEVICE explicit Vector(const T s);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE explicit Vector(const U data[3]);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<3, U>& vec);

    // vec.w == vec[3] ignored.
    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE explicit Vector(const Vector<4, U>& vec);

#ifdef __CUDACC__
    // float must be convertible to T.
    GRACE_HOST_DEVICE Vector(const float3& xyz);

    // double must be convertible to T.
    GRACE_HOST_DEVICE Vector(const double3& xyz);

    // w ignored.
    // float must be convertible to T.
    GRACE_HOST_DEVICE explicit Vector(const float4& xyzw);

    // w ignored.
    // double must be convertible to T.
    GRACE_HOST_DEVICE explicit Vector(const double4& xyzw);
#endif

    template <typename U>
    GRACE_HOST_DEVICE Vector& operator=(const Vector<3, U>& rhs);

    // Always safe, but will always incur some runtime overhead when i is not
    // known at compile-time. When i is known at compile time, likely no
    // overhead.
    GRACE_HOST_DEVICE T& operator[](int i);
    GRACE_HOST_DEVICE const T& operator[](int i) const;
};

template <typename T>
struct Vector<4, T> : detail::vector_base<4, sizeof(T), T>
{
    typedef T value_type;

    T x;
    T y;
    T z;
    T w;

    // Zero-initialized.
    GRACE_HOST_DEVICE Vector();

    GRACE_HOST_DEVICE Vector(const T x, const T y, const T z, const T w);

    GRACE_HOST_DEVICE explicit Vector(const T s);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE explicit Vector(const U data[4]);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<4, U>& vec);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE Vector(const Vector<3, U>& vec, const U w);

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

    // Always safe, but will always incur some runtime overhead when i is not
    // known at compile-time. When i is known at compile time, likely no
    // overhead.
    GRACE_HOST_DEVICE T& operator[](int i);
    GRACE_HOST_DEVICE const T& operator[](int i) const;
};


//
// Comparison operations
//

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
bool operator==(const Vector<Dims, T>& lhs, const Vector<Dims, T>& rhs);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
bool operator!=(const Vector<Dims, T>& lhs, const Vector<Dims, T>& rhs);


//
// Vector operations
//

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


//
// Vector-vector operations
//

// Cross-product, angular and great-circle operators only definied for
// length-three vectors.
template <typename T>
GRACE_HOST_DEVICE
Vector<3, T> cross(const Vector<3, T>& u, const Vector<3, T>& v);

// In radians.
template <typename T>
GRACE_HOST_DEVICE
T angular_separation(const Vector<3, T>& u, const Vector<3, T>& v);

template <typename T>
GRACE_HOST_DEVICE
T great_circle_distance(const Vector<3, T>& u, const Vector<3, T>& v,
                        const T radius);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T dot(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> max(const Vector<Dims, T>& u, const Vector<Dims, T>& v);

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> min(const Vector<Dims, T>& u, const Vector<Dims, T>& v);


//
// Vector-vector arithmetic
//

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


//
// Vector-scalar arithmetic
//

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


//
// Scalar-vector arithmetic
//

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

} // namespace grace

#include "grace/detail/vector_math-inl.h"
#include "grace/detail/vector3-inl.h"
#include "grace/detail/vector4-inl.h"
