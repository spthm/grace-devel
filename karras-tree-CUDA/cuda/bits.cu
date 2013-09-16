#include "../types.h"
#include "bits.cuh"

namespace grace {

namespace gpu {

template <typename UInteger>
__device__ UInteger32 space_by_1(UInteger x) {
    // Mask higher bits and cast to 32-bit unsigned integer.
    x = (UInteger32) x & 1023u;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

template <typename UInteger>
__device__ UInteger32 space_by_2(UInteger x) {
    x = (UInteger32) x & 1023u;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;
    return x;
}

template <typename UInteger>
__device__ UInteger bit_prefix(UInteger a, UInteger b) {
    return __clz(a^b);
}

} // namespace gpu

} // namespace grace
