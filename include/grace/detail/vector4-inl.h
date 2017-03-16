#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.
#include <grace/error.h>

namespace grace {

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector() : x(0), y(0), z(0), w(0) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const T x, const T y, const T z, const T w) : x(x), y(y), z(z), w(w) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const T s) : x(s), y(s), z(s), w(s) {}

// U must be convertible to T.
template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const U data[4]) : x(data[0]), y(data[1]), z(data[2]), w(data[3]) {}

// U must be convertible to T.
template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const Vector<4, U>& vec) : x(vec.x), y(vec.y), z(vec.z), w(vec.w) {}

// U must be convertible to T.
template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const Vector<3, U>& vec, const U w) : x(vec.x), y(vec.y), z(vec.z), w(w) {}

#ifdef __CUDACC__
// float must be convertible to T.
template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const float3& xyz, const float w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

// double must be convertible to T.
template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const double3& xyz, const double w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

// float must be convertible to T.
template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const float4& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}

// double must be convertible to T.
template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const double4& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}
#endif

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>& Vector<4, T>::operator=(const Vector<4, U>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<4, T>::operator[](int i)
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
        case 3: return this->w;
    }

    GRACE_ASSERT(0, VECTOR4_INVALID_INDEX_ACCESS);
    return this->x;
}

template <typename T>
GRACE_HOST_DEVICE
const T& Vector<4, T>::operator[](int i) const
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
        case 3: return this->w;
    }

    GRACE_ASSERT(0, VECTOR4_INVALID_INDEX_ACCESS);
    return this->x;
}

} // namespace grace
