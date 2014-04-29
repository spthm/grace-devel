#pragma once

#include <thrust/device_vector.h>

#include "../kernel_config.h"
#include "../utils.cuh"
#include "bits.cuh"

namespace grace {

//-----------------------------------------------------------------------------
// Helper functions (host-compatible) for generating morton keys
//-----------------------------------------------------------------------------

// 30-bit keys.
__host__ __device__ uinteger32 morton_key(const uinteger32 x,
                                          const uinteger32 y,
                                          const uinteger32 z)
{
    return space_by_two_10bit(z) << 2 | space_by_two_10bit(y) << 1 | space_by_two_10bit(x);
}

// 63-bit keys.
__host__ __device__ uinteger64 morton_key(const uinteger64 x,
                                          const uinteger64 y,
                                          const uinteger64 z)
{
    return space_by_two_21bit(z) << 2 | space_by_two_21bit(y) << 1 | space_by_two_21bit(x);
}

// 30-bit keys from floats.  Assumes floats lie in (0, 1)!
__host__ __device__ uinteger32 morton_key(const float x,
                                          const float y,
                                          const float z)
{
    unsigned int span = (1u << 10) - 1;
    return morton_key((uinteger32) (span*x),
                      (uinteger32) (span*y),
                      (uinteger32) (span*z));

}

// 63-bit keys from doubles.  Assumes doubles lie in (0, 1)!
__host__ __device__ uinteger64 morton_key(const double x,
                                          const double y,
                                          const double z)
{
    unsigned int span = (1u << 21) - 1;
    return morton_key((uinteger64) (span*x),
                      (uinteger64) (span*y),
                      (uinteger64) (span*z));

}

namespace gpu {

//-----------------------------------------------------------------------------
// CUDA kernels for generating morton keys
//-----------------------------------------------------------------------------

template <typename UInteger, typename Float4>
__global__ void morton_keys_kernel(UInteger* keys,
                                   const Float4* xyzr,
                                   const size_t n_points,
                                   const float3 scale)
{
    uinteger32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n_points) {
        UInteger x = (UInteger) (scale.x * xyzr[tid].x);
        UInteger y = (UInteger) (scale.y * xyzr[tid].y);
        UInteger z = (UInteger) (scale.z * xyzr[tid].z);

        keys[tid] = morton_key(x, y, z);

        tid += blockDim.x * gridDim.x;
    }
    return;
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for morton key kernels
//-----------------------------------------------------------------------------

template <typename UInteger, typename Float4>
void morton_keys(thrust::device_vector<UInteger>& d_keys,
                 const thrust::device_vector<Float4>& d_points,
                 const float3 AABB_top,
                 const float3 AABB_bot)
{
    unsigned int span = CHAR_BIT * sizeof(UInteger) > 32 ?
                            ((1u << 21) - 1) : ((1u << 10) - 1);
    float3 scale = make_float3(span / (AABB_top.x - AABB_bot.x),
                               span / (AABB_top.y - AABB_bot.y),
                               span / (AABB_top.z - AABB_bot.z));
    size_t n_points = d_points.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_points + MORTON_THREADS_PER_BLOCK-1)
                                        / MORTON_THREADS_PER_BLOCK));

    d_keys.resize(n_points);
    gpu::morton_keys_kernel<<<blocks,MORTON_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_points.data()),
        n_points,
        scale);
}

template <typename UInteger, typename Float4>
void morton_keys(thrust::device_vector<UInteger>& d_keys,
                 const thrust::device_vector<Float4>& d_points)
{
    float min_x, max_x;
    grace::min_max_x(&min_x, &max_x, d_points);

    float min_y, max_y;
    grace::min_max_y(&min_y, &max_y, d_points);

    float min_z, max_z;
    grace::min_max_z(&min_z, &max_z, d_points);

    float3 bot = make_float3(min_x, min_y, min_z);
    float3 top = make_float3(max_x, max_y, max_z);

    morton_keys(d_keys, d_points, top, bot);
}

} // namespace grace
