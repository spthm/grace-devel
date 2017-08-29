#pragma once

// No grace/morton.h include.
// This should only ever be included by grace/morton.h.

#include "grace/config.h"
#include "grace/types.h"
#include "grace/detail/bits.h"

namespace grace {

GRACE_HOST_DEVICE uinteger32 morton_key(
    const uinteger32 x,
    const uinteger32 y,
    const uinteger32 z)
{
    return detail::space_by_two_10bit(z) << 2 | detail::space_by_two_10bit(y) << 1 | detail::space_by_two_10bit(x);
}

GRACE_HOST_DEVICE uinteger64 morton_key(
    const uinteger64 x,
    const uinteger64 y,
    const uinteger64 z)
{
    return detail::space_by_two_21bit(z) << 2 | detail::space_by_two_21bit(y) << 1 | detail::space_by_two_21bit(x);
}

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
