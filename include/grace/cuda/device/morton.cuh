#pragma once

#include "grace/cuda/device/bits.cuh"

#include "grace/types.h"

namespace grace {

namespace morton {

//-----------------------------------------------------------------------------
// Helper functions (host-compatible) for generating morton keys
//-----------------------------------------------------------------------------

// 30-bit keys.
GRACE_HOST_DEVICE uinteger32 morton_key(
    const uinteger32 x,
    const uinteger32 y,
    const uinteger32 z)
{
    return bits::space_by_two_10bit(z) << 2 | bits::space_by_two_10bit(y) << 1 | bits::space_by_two_10bit(x);
}

// 63-bit keys.
GRACE_HOST_DEVICE uinteger64 morton_key(
    const uinteger64 x,
    const uinteger64 y,
    const uinteger64 z)
{
    return bits::space_by_two_21bit(z) << 2 | bits::space_by_two_21bit(y) << 1 | bits::space_by_two_21bit(x);
}

// 30-bit keys from floats.  Assumes floats lie in (0, 1)!
GRACE_HOST_DEVICE uinteger32 morton_key(
    const float x,
    const float y,
    const float z)
{
    unsigned int span = (1u << 10) - 1;
    return morton_key(static_cast<uinteger32>(span * x),
                      static_cast<uinteger32>(span * y),
                      static_cast<uinteger32>(span * z));

}

// 63-bit keys from doubles.  Assumes doubles lie in (0, 1)!
GRACE_HOST_DEVICE uinteger64 morton_key(
    const double x,
    const double y,
    const double z)
{
    unsigned int span = (1u << 21) - 1;
    return morton_key(static_cast<uinteger64>(span * x),
                      static_cast<uinteger64>(span * y),
                      static_cast<uinteger64>(span * z));

}

} // namespace morton

} // namespace grace
