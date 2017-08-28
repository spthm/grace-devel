#pragma once

#include "grace/config.h"

/* This file provides device-compatible implementations of some of the standard
 * library header <limits>. This allows generic host- and device-compatible
 * code to be written that will compile (for the host) when CUDA/Thrust is not
 * available.
 */

namespace grace {

//
// Host-side, not CUDA-compatible.
//

template <typename T>
struct numeric_limits
{
    // TODO: C++11, these should be static constexpr.
    GRACE_HOST static T epsilon();
    GRACE_HOST static T infinity();
    GRACE_HOST static T lowest();
    GRACE_HOST static T min();
    GRACE_HOST static T max();
};


//
// CUDA-compatible specializations
//

template <>
struct numeric_limits<unsigned int>
{
    GRACE_HOST_DEVICE static unsigned int lowest();
    GRACE_HOST_DEVICE static unsigned int min();
    GRACE_HOST_DEVICE static unsigned int max();
};

template <>
struct numeric_limits<int>
{
    GRACE_HOST_DEVICE static int lowest();
    GRACE_HOST_DEVICE static int min();
    GRACE_HOST_DEVICE static int max();
};

template <>
struct numeric_limits<float>
{
    GRACE_HOST_DEVICE static float epsilon();
    GRACE_HOST_DEVICE static float infinity();
    GRACE_HOST_DEVICE static float lowest();
    GRACE_HOST_DEVICE static float min();
    GRACE_HOST_DEVICE static float max();
};

template <>
struct numeric_limits<double>
{
    GRACE_HOST_DEVICE static double epsilon();
    GRACE_HOST_DEVICE static double infinity();
    GRACE_HOST_DEVICE static double lowest();
    GRACE_HOST_DEVICE static double min();
    GRACE_HOST_DEVICE static double max();
};


} // namespace grace

#include "grace/generic/detail/limits-inl.h"
