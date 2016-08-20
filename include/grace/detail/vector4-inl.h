#pragma once

#include "grace/vector.h"

namespace grace {

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector() : x(0), y(0), z(0), w(0) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const T x, const T y, const T z, const T w) :
    x(x), y(y), z(z), w(w) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const T s) : x(s), y(s), z(s), w(s) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const U data[4]) :
    x(data[0]), y(data[1]), z(data[2]), w(data[3]) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const Vector<4, U>& vec) :
    x(vec.x), y(vec.y), z(vec.y), w(vec.r) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const Vector<3, U>& vec, const U w) :
    x(vec.x), y(vec.y), z(vec.z), w(w) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const Sphere<U>& s) :
    x(s.x), y(s.y), z(s.z), w(s.r) {}

#ifdef __CUDACC__
template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const float3& xyz, const float w) :
    x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const double3& xyz, const double w) :
    x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const float4& xyzw) :
    x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<4, T>::Vector(const double4& xyzw) :
    x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}
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
T* Vector<4, T>::data()
{
    // Generally unsafe, but detail::VectorMember<4, T> ensures it is safe iff
    // T is a 1, 2, 4 or 8-byte type.
    return &(this->x);
}

template <typename T>
GRACE_HOST_DEVICE
const T* Vector<4, T>::data() const
{
    return const_cast<const T*>(this->data());
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<4, T>::operator[](size_t i)
{
    return this->data()[i];
}

template <typename T>
GRACE_HOST_DEVICE
const T& vector<4, T>::operator[](size_t i) const
{
    // Overloads to const data().
    return this->data()[i];
}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>& Vector<4, T>::vec3()
{
    return *reinterpret_cast<Vector<3, T>*>(this->data());
}

template <typename T>
GRACE_HOST_DEVICE
const Vector<3, T>& Vector<4, T>::vec3() const
{
    // Overloads to const data().
    return *reinterpret_cast<const Vector<3, T>*>(this->data());
}

} // namespace grace
