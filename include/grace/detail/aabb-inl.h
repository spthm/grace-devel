#pragma once

// No grace/aabb.h include.
// This should only ever be included by grace/aabb.h.

namespace grace {

//
// Constructors and member functions
//

template <typename T>
GRACE_HOST_DEVICE AABB<T>::AABB()
    : min((T)-0.5, (T)-0.5, (T)-0.5), max((T)0.5, (T)0.5, (T)0.5) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE AABB<T>::AABB(const Vector<3, U>& min, const Vector<3, U>& max)
    : min(min), max(max) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE AABB<T>::AABB(const U min[3], const U max[3])
    : min(min), max(max) {}

template <typename T>
template <typename U>
GRACE_HOST_DEVICE AABB<T>::AABB(const AABB<U>& other)
    : min(other.min), max(other.max) {}

#ifdef __CUDACC__
template <typename T>
GRACE_HOST_DEVICE AABB<T>::AABB(const float3& min, const float3& max)
    : min(min), max(max) {}

template <typename T>
GRACE_HOST_DEVICE AABB<T>::AABB(const double3& min, const double3& max)
    : min(min), max(max) {}
#endif

template <typename T>
GRACE_HOST_DEVICE T AABB<T>::area() const
{
    const Vector<3, T> s = size();
    return T(2) * (s.x * s.y + s.x * s.z + s.y * s.z);
}

template <typename T>
GRACE_HOST_DEVICE Vector<3, T> AABB<T>::center() const
{
    return T(0.5) * (max + min);
}

template <typename T>
GRACE_HOST_DEVICE Vector<3, T> AABB<T>::size() const
{
    return max - min;
}

template <typename T>
GRACE_HOST_DEVICE void AABB<T>::scale(const T s)
{
    const Vector<3, T> c = center();

    min = min - c; max = max - c;

    min = min * s; max = max * s;

    min = min + c; max = max + c;
}

template <typename T>
GRACE_HOST_DEVICE void AABB<T>::scale(const Vector<3, T>& vec)
{
    const Vector<3, T> c = center();

    min = min - c; max = max - c;

    min = min * vec; max = max * vec;

    min = min + c; max = max + c;
}

template <typename T>
GRACE_HOST_DEVICE void AABB<T>::translate(const Vector<3, T>& vec)
{
    min = min + vec;
    max = max + vec;
}


//
// Comparison operations
//

template <typename T>
GRACE_HOST_DEVICE
bool operator==(const AABB<T>& lhs, const AABB<T>& rhs)
{
    return lhs.min == rhs.min && lhs.max == rhs.max;
}

template <typename T>
GRACE_HOST_DEVICE
bool operator!=(const AABB<T>& lhs, const AABB<T>& rhs)
{
    return lhs.min != rhs.min || lhs.max != rhs.max;
}

} // namespace grace
