#pragma once

#include "grace/config.h"
#include "grace/types.h"

namespace grace {

// 30-bit keys.
GRACE_HOST_DEVICE uinteger32 morton_key(
    const uinteger32 x,
    const uinteger32 y,
    const uinteger32 z);

// 63-bit keys.
GRACE_HOST_DEVICE uinteger64 morton_key(
    const uinteger64 x,
    const uinteger64 y,
    const uinteger64 z);

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
    const float z);

// 30-bit keys from doubles.  Assumes doubles lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger32 morton_key<uinteger32, double>(
    const double x,
    const double y,
    const double z);

// 63-bit keys from floats.  Assumes floats lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger64 morton_key<uinteger64, float>(
    const float x,
    const float y,
    const float z);

// 63-bit keys from doubles.  Assumes doubles lie in (0, 1)!
template <>
GRACE_HOST_DEVICE uinteger64 morton_key<uinteger64, double>(
    const double x,
    const double y,
    const double z);

} // namespace grace

#include "grace/detail/morton-inl.h"
