#pragma once

#include "../types.h"

namespace grace {

namespace gpu {

template <typename UInteger, typename Float>
class morton_key_functor {};

__host__ __device__ UInteger32 morton_key_30bit(UInteger32 x,
                                                UInteger32 y,
                                                UInteger32 z);

__host__ __device__ UInteger64 morton_key_63bit(UInteger32 x,
                                                UInteger32 y,
                                                UInteger32 z);

} // namespace gpu

} // namespace grace
