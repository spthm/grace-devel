#pragma once

#include "grace/config.h"
#include "grace/types.h"
#include "grace/generic/bits.h"

namespace grace {

//-----------------------------------------------------------------------------
// Helper functions (host-compatible) for generating morton keys
//-----------------------------------------------------------------------------

// 30-bit keys.
GRACE_HOST_DEVICE uinteger32 morton_key(
    const uinteger32 x,
    const uinteger32 y,
    const uinteger32 z)
{
    return detail::space_by_two_10bit(z) << 2 | detail::space_by_two_10bit(y) << 1 | detail::space_by_two_10bit(x);
}

// 63-bit keys.
GRACE_HOST_DEVICE uinteger64 morton_key(
    const uinteger64 x,
    const uinteger64 y,
    const uinteger64 z)
{
    return detail::space_by_two_21bit(z) << 2 | detail::space_by_two_21bit(y) << 1 | detail::space_by_two_21bit(x);
}

// Templated so we can (effectively) overload based on return type.
// Only specific explicit instantiations are provided:
//   <uinteger32, float>
//   <uinteger32, double>
//   <uinteger64, float>
//   <uinteger64, double>
template <typename KeyType, typename Real>
GRACE_HOST_DEVICE KeyType morton_key(const Real, const Real, const Real);

// 30-bit keys from floats.  Assumes floats lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger32 morton_key<uinteger32, float>(
    const float x,
    const float y,
    const float z)
{
    unsigned int span = (1u << 10) - 1;
    return morton_key(static_cast<uinteger32>(span * x),
                      static_cast<uinteger32>(span * y),
                      static_cast<uinteger32>(span * z));
}

// 30-bit keys from doubles.  Assumes doubles lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger32 morton_key<uinteger32, double>(
    const double x,
    const double y,
    const double z)
{
    unsigned int span = (1u << 10) - 1;
    return morton_key(static_cast<uinteger32>(span * x),
                      static_cast<uinteger32>(span * y),
                      static_cast<uinteger32>(span * z));
}

// 63-bit keys from floats.  Assumes floats lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger64 morton_key<uinteger64, float>(
    const float x,
    const float y,
    const float z)
{
    unsigned int span = (1u << 21) - 1;
    return morton_key(static_cast<uinteger64>(span * x),
                      static_cast<uinteger64>(span * y),
                      static_cast<uinteger64>(span * z));
}

// 63-bit keys from doubles.  Assumes doubles lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger64 morton_key<uinteger64, double>(
    const double x,
    const double y,
    const double z)
{
    unsigned int span = (1u << 21) - 1;
    return morton_key(static_cast<uinteger64>(span * x),
                      static_cast<uinteger64>(span * y),
                      static_cast<uinteger64>(span * z));
}

} // namespace grace
