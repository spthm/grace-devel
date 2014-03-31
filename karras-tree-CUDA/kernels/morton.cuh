#pragma once

#include <thrust/device_vector.h>

#include "../kernel_config.h"
#include "../types.h"
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

template <typename UInteger, typename Float>
__global__ void morton_keys_kernel(const Float* xs,
                                   const Float* ys,
                                   const Float* zs,
                                   UInteger* keys,
                                   const size_t n_keys,
                                   const Vector3<Float> scale)
{
    uinteger32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n_keys) {
        UInteger x = (UInteger) (scale.x * xs[tid]);
        UInteger y = (UInteger) (scale.y * ys[tid]);
        UInteger z = (UInteger) (scale.z * zs[tid]);

        keys[tid] = morton_key(x, y, z);

        tid += blockDim.x * gridDim.x;
    }
    return;
}

template <typename UInteger, typename Float4>
__global__ void morton_keys_kernel(const Float4* xyzr,
                                   UInteger* keys,
                                   const size_t n_keys,
                                   const Float4 scale)
{
    uinteger32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n_keys) {
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

template <typename UInteger, typename Float>
void morton_keys(const thrust::device_vector<Float>& d_xs,
                 const thrust::device_vector<Float>& d_ys,
                 const thrust::device_vector<Float>& d_zs,
                 thrust::device_vector<UInteger>& d_keys,
                 const Vector3<Float>& AABB_bottom,
                 const Vector3<Float>& AABB_top)
{
    unsigned int span = CHAR_BIT * sizeof(UInteger) > 32 ?
                            ((1u << 21) - 1) : ((1u << 10) - 1);
    Vector3<Float> scale((Float)span / (AABB_top.x - AABB_bottom.x),
                         (Float)span / (AABB_top.y - AABB_bottom.y),
                         (Float)span / (AABB_top.z - AABB_bottom.z));
    size_t n_keys = d_xs.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_keys + MORTON_THREADS_PER_BLOCK-1)
                                        / MORTON_THREADS_PER_BLOCK));

    d_keys.resize(n_keys);
    gpu::morton_keys_kernel<<<blocks,MORTON_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_xs.data()),
        thrust::raw_pointer_cast(d_ys.data()),
        thrust::raw_pointer_cast(d_zs.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys,
        scale);
}

template <typename UInteger, typename Float4>
void morton_keys(const thrust::device_vector<Float4>& d_spheres_xyzr,
                 thrust::device_vector<UInteger>& d_keys,
                 const Float4 AABB_bottom,
                 const Float4 AABB_top)
{
    unsigned int span = CHAR_BIT * sizeof(UInteger) > 32 ?
                            ((1u << 21) - 1) : ((1u << 10) - 1);
    Float4 scale = make_float4(span / (AABB_top.x - AABB_bottom.x),
                               span / (AABB_top.y - AABB_bottom.y),
                               span / (AABB_top.z - AABB_bottom.z),
                               0);
    size_t n_keys = d_spheres_xyzr.size();

    int blocks = min(MAX_BLOCKS, (int) ((n_keys + MORTON_THREADS_PER_BLOCK-1)
                                        / MORTON_THREADS_PER_BLOCK));

    d_keys.resize(n_keys);
    gpu::morton_keys_kernel<<<blocks,MORTON_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_spheres_xyzr.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        n_keys,
        scale);
}

} // namespace grace
