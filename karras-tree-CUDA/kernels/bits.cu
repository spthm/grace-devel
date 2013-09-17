#include "../types.h"
#include "bits.cuh"

namespace grace {

namespace gpu {

__host__ __device__ UInteger32 space_by_two_10bit(UInteger32 x) {
    // Mask high bits away first.
    x &= (1u << 10) - 1;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;
    return x;
}

// Courtesy of http://stackoverflow.com/a/18529061/927046
__host__ __device__ UInteger32 space_by_two_21bit(UInteger32 x) {
    x &= (1u << 21) - 1;
    x = (x | x << 32) & 0x001f00000000ffff;
    x = (x | x << 16) & 0x001f0000ff0000ff;
    x = (x | x <<  8) & 0x100f00f00f00f00f;
    x = (x | x <<  4) & 0x10c30c30c30c30c3;
    x = (x | x <<  2) & 0x1249249249249249;
    return x;
}

template <typename UInteger>
__device__ UInteger bit_prefix(UInteger a, UInteger b) {
    return __clz(a^b);

template UInteger32 bit_prefix<UInteger32>(UInteger32 a, UInteger32 b);
template UInteger64 bit_prefix<UInteger64>(UInteger64 a, UInteger64 b);
}

} // namespace gpu

} // namespace grace
