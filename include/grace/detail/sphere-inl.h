#pragma once

#include "grace/sphere.h"

namespace grace {

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere() : x(0), y(0), z(0), r(1) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(T x, T y, T z, T r) : x(x), y(y), z(z), r(r) {}

#ifdef __CUDACC__
template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(float4 xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), r(xyzw.w) {}

template <typename T>
GRACE_HOST_DEVICE Sphere<T>::Sphere(double4 xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), r(xyzw.w) {}
#endif

} // namespace grace
