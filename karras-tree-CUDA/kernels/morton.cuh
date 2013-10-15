#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "bits.cuh"

namespace grace {

// 30-bit keys.
__host__ __device__ UInteger32 morton_key(const UInteger32& x,
                                          const UInteger32& y,
                                          const UInteger32& z) {
    return space_by_two_10bit(z) << 2 | space_by_two_10bit(y) << 1 | space_by_two_10bit(x);
}

// 63-bit keys.
__host__ __device__ UInteger64 morton_key(const UInteger64& x,
                                          const UInteger64& y,
                                          const UInteger64& z) {
    return space_by_two_21bit(z) << 2 | space_by_two_21bit(y) << 1 | space_by_two_21bit(x);
}

namespace gpu {

template <typename UInteger, typename Float>
__global__ void morton_keys_kernel(const Float* xs,
                                   const Float* ys,
                                   const Float* zs,
                                   UInteger* keys,
                                   const UInteger32 n_keys,
                                   const Vector3<Float> scale)
{
    UInteger32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n_keys) {
        UInteger x = (UInteger) scale.x * xs[tid];
        UInteger y = (UInteger) scale.y * ys[tid];
        UInteger z = (UInteger) scale.z * zs[tid];

        keys[tid] = morton_key(x, y, z);

        tid += blockDim.x * gridDim.x;
    }
    return;
}

} // namespace gpu

template <typename UInteger, typename Float>
void morton_keys(const thrust::device_vector<Float>& d_xs,
                 const thrust::device_vector<Float>& d_ys,
                 const thrust::device_vector<Float>& d_zs,
                 thrust::device_vector<UInteger>& d_keys;
                 const Vector3<Float>& AABB_bottom,
                 const Vector3<Float>& AABB_top)
{
    // Should be optimized away by the compiler.
    unsigned int span = CHAR_BIT * sizeof(UInteger) > 32 ?
                            ((1u << 21) - 1) : ((1u << 10) - 1);
    Vector3<Float> scale((Float)span / (AABB_top.x - AABB_bottom.x)
                         (Float)span / (AABB_top.y - AABB_bottom.y)
                         (Float)span / (AABB_top.z - AABB_bottom.z));
    UInteger32 n_keys = d_xs.size();

    scale.x = span / (AABB_top.x - AABB_bottom.x);
    scale.y = span / (AABB_top.y - AABB_bottom.y);
    scale.z = span / (AABB_top.z - AABB_bottom.z);

    int blocks = min(MAX_BLOCKS, (n_leaves + THREADS_PER_BLOCK-1)
                                  / THREADS_PER_BLOCK);

    d_keys.resize(n_keys);
    gpu::morton_keys_kernel<<<blocks, MAX_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_xs.data()),
        thrust::raw_pointer_cast(d_ys.data()),
        thrust::raw_pointer_cast(d_zs.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys,
        scale);
}

} // namespace grace
