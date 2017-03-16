#pragma once

#include "grace/types.h"
#include "grace/vector.h"

namespace grace {

template <typename T>
struct GRACE_ALIGNAS(16) AABB
{
    typedef T value_type;
    typedef Vector<3, T> vec_type;

    Vector<3, T> min;
    Vector<3, T> max;

    // Unit box at origin.
    GRACE_HOST_DEVICE AABB();

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE AABB(const Vector<3, U>& min, const Vector<3, U>& max);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE AABB(const U min[3], const U max[3]);

    // U must be convertible to T.
    template <typename U>
    GRACE_HOST_DEVICE AABB(const AABB<U>& other);

#ifdef __CUDACC__
    // Unit box at origin.
    // float must be convertible to T.
    GRACE_HOST_DEVICE AABB(const float3& min, const float3& max);

    // Unit box at origin.
    // double must be convertible to T.
    GRACE_HOST_DEVICE AABB(const double3& min, const double3& max);
#endif

    GRACE_HOST_DEVICE T area() const;

    GRACE_HOST_DEVICE Vector<3, T> center() const;

    GRACE_HOST_DEVICE Vector<3, T> size() const;

    GRACE_HOST_DEVICE void scale(const T s);
    GRACE_HOST_DEVICE void scale(const Vector<3, T>& vec);

    GRACE_HOST_DEVICE void translate(const Vector<3, T>& vec);
};


//
// Comparison operations
//

template <typename T>
GRACE_HOST_DEVICE
bool operator==(const AABB<T>& lhs, const AABB<T>& rhs);

template <typename T>
GRACE_HOST_DEVICE
bool operator!=(const AABB<T>& lhs, const AABB<T>& rhs);

} // namespace grace

#include "grace/detail/aabb-inl.h"
