#pragma once

#include <cmath>
#include <iostream>
#include <bitset>

#include "../types.h"

namespace grace {

template <typename UInteger>
__host__ __device__ UInteger32 space_by_two_10bit(UInteger x) {
    // Mask high bits away first, and ensure we have enough bits.
    UInteger32 x_32 = (UInteger32)x & ((1u << 10) - 1);
    x_32 = (x_32 | (x_32 << 16)) & 0x030000FF;
    x_32 = (x_32 | (x_32 <<  8)) & 0x0300F00F;
    x_32 = (x_32 | (x_32 <<  4)) & 0x030C30C3;
    x_32 = (x_32 | (x_32 <<  2)) & 0x09249249;
    return x_32;
}

// Courtesy of http://stackoverflow.com/a/18529061/927046
template <typename UInteger>
__host__ __device__ UInteger64 space_by_two_21bit(UInteger x) {
    // This spaced integer requires 3*21 = 63 bits
    UInteger64 x_64 = (UInteger64)x & ((1u << 21) - 1);
    x_64 = (x_64 | x_64 << 32) & 0x001f00000000ffff;
    x_64 = (x_64 | x_64 << 16) & 0x001f0000ff0000ff;
    x_64 = (x_64 | x_64 <<  8) & 0x100f00f00f00f00f;
    x_64 = (x_64 | x_64 <<  4) & 0x10c30c30c30c30c3;
    x_64 = (x_64 | x_64 <<  2) & 0x1249249249249249;
    return x_64;
}

template <typename UInteger>
__host__ __device__ UInteger bit_prefix(const UInteger a, const UInteger b) {
    unsigned int n_bits = CHAR_BIT * sizeof(UInteger);
    UInteger x_or = a ^ b;
    if (x_or > 0)
        // Count leading zeros of the xor.
        return n_bits - 1 - UInteger(floor(log2((float)x_or)));
    else
        // a == b
        return n_bits;
}

namespace gpu {

// TODO: Rename this to e.g. bit_prefix_length
template <typename UInteger>
__device__ UInteger bit_prefix(const UInteger a, const UInteger b) {
    // sizeof is a compile time operator, so the conditional return should be
    // optimized away at compile time.
    return (CHAR_BIT * sizeof(UInteger)) > 32 ? __clzll(a^b) : __clz(a^b);
}

} // namespace gpu

} // namespace grace