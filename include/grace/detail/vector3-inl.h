#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.
#include <grace/error.h>

namespace grace {

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector() : x(0), y(0), z(0) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const T x, const T y, const T z) : x(x), y(y), z(z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const T s) : x(s), y(s), z(s) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const U data[3]) : x(data[0]), y(data[1]), z(data[2]) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const Vector<3, U>& vec) : x(vec.x), y(vec.y), z(vec.z) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const Vector<4, U>& vec) : x(vec.x), y(vec.y), z(vec.z) {}

#ifdef __CUDACC__
template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const float3& xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const double3& xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const float4& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const double4& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z) {}
#endif

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>& Vector<3, T>::operator=(const Vector<3, U>& rhs)
{
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<3, T>::operator[](int i)
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
    }

    GRACE_ASSERT(0, VECTOR3_INVALID_INDEX_ACCESS);
    return this->x;
}

template <typename T>
GRACE_HOST_DEVICE
const T& Vector<3, T>::operator[](int i) const
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
    }

    GRACE_ASSERT(0, VECTOR3_INVALID_INDEX_ACCESS);
    return this->x;
}

} // namespace grace
