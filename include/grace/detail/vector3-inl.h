#pragma once

#include "grace/vector.h"

namespace grace {

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector() : x(0), y(0), z(0) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const T x, const T y, const T z) :
    x(x), y(y), z(z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const T s) : x(s), y(s), z(s) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const U data[3]) :
    x(data[0]), y(data[1]), z(data[2]) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const Vector<3, U>& vec) :
    x(vec.x), y(vec.y), z(vec.y) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const Vector<4, U>& vec) :
    x(vec.x), y(vec.y), z(vec.z) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const Sphere<U>& s) :
    x(s.x), y(s.y), z(s.z) {}

#ifdef __CUDACC__
template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const float3& xyz) :
    x(xyz.x), y(xyz.y), z(xyz.z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const double3& xyz) :
    x(xyz.x), y(xyz.y), z(xyz.z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const float4& xyzw) :
    x(xyzw.x), y(xyzw.y), z(xyzw.z) {}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>::Vector(const double4& xyzw) :
    x(xyzw.x), y(xyzw.y), z(xyzw.z) {}
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
T* Vector<3, T>::get()
{
    // Element order in memory guaranteed identical to order of declaration.
    // However, depending on the architecture and compiler, this may be unsafe
    // for sizeof(T) < 4: there may be padding after each element.
    return &(this->x);
}

template <typename T>
GRACE_HOST_DEVICE
const T* Vector<3, T>::get() const
{
    return const_cast<const T*>(this->get());
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<3, T>::operator[](size_t i)
{
    return this->get()[i];
}

template <typename T>
GRACE_HOST_DEVICE
const T& vector<3, T>::operator[](size_t i) const
{
    // Overloads to const get().
    return this->get()[i];
}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>& Vector<3, T>::vec3()
{
    return *reinterpret_cast<Vector<3, T>*>(this->get());
}

template <typename T>
GRACE_HOST_DEVICE
const Vector<3, T>& Vector<3, T>::vec3() const
{
    // Overloads to const get().
    return *reinterpret_cast<const Vector<3, T>*>(this->get());
}

} // namespace grace
