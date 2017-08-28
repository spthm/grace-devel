#pragma once

#include "grace/config.h"
#include "grace/types.h"

#include <cmath>
#include <iostream>
#include <bitset>

namespace grace {

template <typename T>
GRACE_HOST_DEVICE int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

namespace detail {

//-----------------------------------------------------------------------------
// Functions for bitwise manipulation.
//-----------------------------------------------------------------------------

template <typename UInteger>
GRACE_HOST_DEVICE uinteger32 space_by_two_10bit(const UInteger x)
{
    // Mask high bits away first, and ensure we have enough bits.
    uinteger32 x_32 = (uinteger32)x & ((1u << 10) - 1);
    x_32 = (x_32 | (x_32 << 16)) & 0x030000FF;
    x_32 = (x_32 | (x_32 <<  8)) & 0x0300F00F;
    x_32 = (x_32 | (x_32 <<  4)) & 0x030C30C3;
    x_32 = (x_32 | (x_32 <<  2)) & 0x09249249;
    return x_32;
}

template <typename UInteger>
GRACE_HOST_DEVICE uinteger64 space_by_two_21bit(const UInteger x)
{
    // This spaced integer requires 3*21 = 63 bits
    uinteger64 x_64 = (uinteger64)x & ((1u << 21) - 1);
    x_64 = (x_64 | x_64 << 32) & 0x001f00000000ffff;
    x_64 = (x_64 | x_64 << 16) & 0x001f0000ff0000ff;
    x_64 = (x_64 | x_64 <<  8) & 0x100f00f00f00f00f;
    x_64 = (x_64 | x_64 <<  4) & 0x10c30c30c30c30c3;
    x_64 = (x_64 | x_64 <<  2) & 0x1249249249249249;
    return x_64;
}

} // namespace detail

} // namespace grace
