#pragma once

// No grace/aabb.h include.
// This should only ever be included by grace/aabb.h.

#include "grace/generic/limits.h"

namespace grace {

//
// Constructors and member functions
//

template <typename T>
GRACE_HOST_DEVICE AABB<T>::AABB()
{
    invalidate();
}

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
GRACE_HOST_DEVICE void AABB<T>::invalidate()
{
    // lowest/max rather than max/min because min, for floats, is the smallest
    // positive value.
    // +/- infinity also an option, but doesn't work for (admittedly weird)
    // integral-value AABBs.
    min = Vector<3, T>(grace::numeric_limits<T>::max(),
                       grace::numeric_limits<T>::max(),
                       grace::numeric_limits<T>::max());
    max = Vector<3, T>(grace::numeric_limits<T>::lowest(),
                       grace::numeric_limits<T>::lowest(),
                       grace::numeric_limits<T>::lowest());

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


//
// Geometric operations
//

template <typename T>
GRACE_HOST_DEVICE
AABB<T> aabb_union(const AABB<T>& lhs, const AABB<T>& rhs)
{
    AABB<T> result;
    result.min = min(lhs.min, rhs.min);
    result.max = max(lhs.max, rhs.max);
    return result;
}

} // namespace grace
