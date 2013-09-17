#pragma once

namespace grace {

namespace gpu {

__host__ __device__ UInteger32 space_by_two_10bit(UInteger32 x);

__host__ __device__ UInteger32 space_by_two_21bit(UInteger32 x);

template <typename UInteger>
__device__ UInteger bit_prefix(UInteger a, UInteger b);

} // namespace gpu

} // namespace grace
