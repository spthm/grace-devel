#pragma once

// No grace/limits.h include.
// This should only ever be included by grace/limits.h.

#include <climits>
#include <cfloat>
#include <cmath>
#include <limits>

namespace grace {

//
// Host-side, not CUDA-compatible.
//

template <typename T>
GRACE_HOST T numeric_limits<T>::epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T>
GRACE_HOST T numeric_limits<T>::infinity()
{
    return std::numeric_limits<T>::infinity();
}

#if __cplusplus >= 201103L
template <typename T>
GRACE_HOST T numeric_limits<T>::lowest()
{
    return std::numeric_limits<T>::lowest();
}
#endif

template <typename T>
GRACE_HOST T numeric_limits<T>::min()
{
    return std::numeric_limits<T>::min();
}

template <typename T>
GRACE_HOST T numeric_limits<T>::max()
{
    return std::numeric_limits<T>::max();
}


//
// CUDA-compatible specializations
//

// unsigned int

GRACE_HOST_DEVICE unsigned int numeric_limits<unsigned int>::lowest()
{
    return 0;
}

GRACE_HOST_DEVICE unsigned int numeric_limits<unsigned int>::min()
{
    return 0;
}

GRACE_HOST_DEVICE unsigned int numeric_limits<unsigned int>::max()
{
    return UINT_MAX;
}

// int

GRACE_HOST_DEVICE int numeric_limits<int>::lowest()
{
    return INT_MIN;
}

GRACE_HOST_DEVICE int numeric_limits<int>::min()
{
    return INT_MIN;
}

GRACE_HOST_DEVICE int numeric_limits<int>::max()
{
    return INT_MAX;
}

// float

GRACE_HOST_DEVICE float numeric_limits<float>::epsilon()
{
    return FLT_EPSILON;
}

GRACE_HOST_DEVICE float numeric_limits<float>::infinity()
{
    return HUGE_VALF;
}

GRACE_HOST_DEVICE float numeric_limits<float>::lowest()
{
    return -FLT_MAX;
}

GRACE_HOST_DEVICE float numeric_limits<float>::min()
{
    return FLT_MIN;
}

GRACE_HOST_DEVICE float numeric_limits<float>::max()
{
    return FLT_MAX;
}

// double

GRACE_HOST_DEVICE double numeric_limits<double>::epsilon()
{
    return DBL_EPSILON;
}

GRACE_HOST_DEVICE double numeric_limits<double>::infinity()
{
    return HUGE_VAL;
}

GRACE_HOST_DEVICE double numeric_limits<double>::lowest()
{
    return -DBL_MAX;
}

GRACE_HOST_DEVICE double numeric_limits<double>::min()
{
    return DBL_MIN;
}

GRACE_HOST_DEVICE double numeric_limits<double>::max()
{
    return DBL_MAX;
}

} // namespace grace
