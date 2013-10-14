#pragma once

#include <thrust/device_vector.h>

#include "../types.h"
#include "bits.cuh"

namespace grace {

template <typename UInteger>
__host__ __device__ UInteger32 morton_key_30bit(const UInteger& x,
                                                const UInteger& y,
                                                const UInteger& z) {
    return space_by_two_10bit(z) << 2 | space_by_two_10bit(y) << 1 | space_by_two_10bit(x);
}

template <typename UInteger>
__host__ __device__ UInteger64 morton_key_63bit(const UInteger& x,
                                                const UInteger& y,
                                                const UInteger& z) {
    return space_by_two_21bit(z) << 2 | space_by_two_21bit(y) << 1 | space_by_two_21bit(x);
}

namespace gpu {

template <typename Float>
__global__ void morton_keys_kernel_30bit(const Float* xs,
                                         const Float* ys,
                                         const Float* zs,
                                         UInteger32* keys,
                                         const UInteger32 n_keys,
                                         const Vector3<Float> scale)
{
    UInteger32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n_keys) {
        UInteger32 x = (UInteger32) scale.x * xs[tid];
        UInteger32 y = (UInteger32) scale.y * ys[tid];
        UInteger32 z = (UInteger32) scale.z * zs[tid];

        keys[tid] = morton_key_30bit(x, y, z);

        tid += blockDim.x * gridDim.x;
    }
    return;
}

template <typename Float>
__global__ void morton_keys_kernel_63bit(const Float* xs,
                                         const Float* ys,
                                         const Float* zs,
                                         UInteger64* keys,
                                         const UInteger64 n_keys,
                                         const Vector3<Float> scale)
{
    UInteger64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n_keys) {
        UInteger64 x = (UInteger64) scale.x * xs[tid];
        UInteger64 y = (UInteger64) scale.y * ys[tid];
        UInteger64 z = (UInteger64) scale.z * zs[tid];

        keys[tid] = morton_key_63bit(x, y, z);

        tid += blockDim.x * gridDim.x;
    }
    return;
}

} // namespace gpu

template <typename UInteger, typename Float>
thrust::device_vector<UInteger> morton_keys(
            const thrust::device_vector<Float>& d_xs,
            const thrust::device_vector<Float>& d_ys,
            const thrust::device_vector<Float>& d_zs,
            const Vector3<Float>& AABB_bottom,
            const Vector3<Float>& AABB_top)
{
    Vector3<Float> scale;
    unsigned int span = (CHAR_BIT * sizeof(UInteger) > 32 ? (1u << 10) : (1u << 21)) - 1;
    UInteger n_keys = d_xs.size();

    scale.x = span / (AABB_top.x - AABB_bottom.x);
    scale.y = span / (AABB_top.y - AABB_bottom.y);
    scale.z = span / (AABB_top.z - AABB_bottom.z);

    int blocks = min(MAX_BLOCKS, (n_leaves + THREADS_PER_BLOCK-1)
                                  / THREADS_PER_BLOCK);

    if (span > 1023) {
        thrust::device_vector<UInteger64> d_keys(n_keys);
        gpu::morton_keys_kernel_30bit<<<blocks, MAX_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_xs.data()),
            thrust::raw_pointer_cast(d_ys.data()),
            thrust::raw_pointer_cast(d_zs.data()),
            thrust::raw_pointer_cast(d_keys.data()),
            n_keys,
            scale);
    }
    else {
        thrust::device_vector<UInteger32> d_keys(n_keys);
        gpu::morton_keys_kernel_63bit<<<blocks, MAX_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_xs.data()),
            thrust::raw_pointer_cast(d_ys.data()),
            thrust::raw_pointer_cast(d_zs.data()),
            thrust::raw_pointer_cast(d_keys.data()),
            n_keys,
            scale);
    }
}

} // namespace grace
