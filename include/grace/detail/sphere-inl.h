#pragma once

// No grace/sphere.h include.
// This should only ever be included by grace/sphere.h.

namespace grace {

//
// Constructors and member functions
//

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere() : x(0), y(0), z(0), r(1) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const T x, const T y, const T z, const T r) : x(x), y(y), z(z), r(r) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const T s) : x(s), y(s), z(s), r(s) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const U data[4]) : x(data[0]), y(data[1]), z(data[2]), r(data[3]) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const Sphere<U>& s) : x(s.x), y(s.y), z(s.y), r(s.r) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const Vector<3, U>& vec) : x(vec.x), y(vec.y), z(vec.z), r(1) {}

template <typename T>
template <typename U, typename S>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const Vector<3, U>& vec, S r) : x(vec.x), y(vec.y), z(vec.z), r(r) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const Vector<4, U>& vec) : x(vec.x), y(vec.y), z(vec.z), r(vec.w) {}

#ifdef __CUDACC__
template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const float3& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), r(1) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const double3& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), r(1) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const float4& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), r(xyzw.w) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(const double4& xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), r(xyzw.w) {}
#endif

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T> Sphere<T>::center() const
{
    return Vector<3, T>(x, y, z);
}


//
// Comparison operations
//

template <typename T>
GRACE_HOST_DEVICE
bool operator==(const Sphere<T>& lhs, const Sphere<T>& rhs)
{
    return lhs.x == rhs.x
        && lhs.y == rhs.y
        && lhs.z == rhs.z
        && lhs.r == rhs.r;
}

template <typename T>
GRACE_HOST_DEVICE
bool operator!=(const Sphere<T>& lhs, const Sphere<T>& rhs)
{
    return lhs.x != rhs.x
        || lhs.y != rhs.y
        || lhs.z != rhs.z
        || lhs.r != rhs.r;
}

} // namespace grace
