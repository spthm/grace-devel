#pragma once

#include <iterator>

#include <thrust/device_vector.h>

#include "aabb.cuh"
#include "../error.h"
#include "../kernel_config.h"
#include "../types.h"
#include "../device/aabb.cuh"
#include "../device/bits.cuh"
#include "../device/morton.cuh"
#include "../util/extrema.cuh"

namespace grace {

namespace morton {

//-----------------------------------------------------------------------------
// CUDA kernel for generating morton keys
//-----------------------------------------------------------------------------

template <typename PrimitiveIter, typename Real3, typename KeyIter,
          typename AABBFunc>
__global__ void morton_keys_kernel(
    PrimitiveIter primitives,
    const size_t N_primitives,
    const Real3 norm_scale,
    KeyIter keys,
    const AABBFunc AABB)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;
    typedef typename std::iterator_traits<KeyIter>::value_type KeyType;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_primitives) {
        float3 centre = AABB::primitive_centroid(primitives[tid], AABB);

        KeyType x = static_cast<KeyType>(norm_scale.x * centre.x);
        KeyType y = static_cast<KeyType>(norm_scale.y * centre.y);
        KeyType z = static_cast<KeyType>(norm_scale.z * centre.z);

        keys[tid] = morton_key(x, y, z);

        tid += blockDim.x * gridDim.x;
    }
    return;
}

//-----------------------------------------------------------------------------
// C-like wrapper for morton key kernel.
//-----------------------------------------------------------------------------

// This functions signature is unlike the morton_key() functions below, which
// is why it has been moved into the 'internal', morton:: namespace
template <typename PrimitiveIter, typename Real3, typename KeyIter,
          typename AABBFunc>
GRACE_HOST void morton_keys(
    PrimitiveIter d_prims_iter,
    const size_t N_primitives,
    const Real3 normalizing_scale,
    KeyIter d_keys_iter,
    const AABBFunc AABB)
{
    int blocks = min(MAX_BLOCKS, (int) ((N_primitives + MORTON_THREADS_PER_BLOCK-1)
                                        / MORTON_THREADS_PER_BLOCK));

    morton_keys_kernel<<<blocks,MORTON_THREADS_PER_BLOCK>>>(
        d_prims_iter,
        N_primitives,
        normalizing_scale,
        d_keys_iter,
        AABB);
    GRACE_KERNEL_CHECK();
}

} // namespace morton


//-----------------------------------------------------------------------------
// C-like wrappers for morton key kernels
//-----------------------------------------------------------------------------

// Wrappers to compute morton keys given the AABB containing all primitives.
// KeyIter's value_type must be an unsigned integer type of at least 32 bits.
template <typename PrimitiveIter, typename Real3, typename KeyIter,
          typename AABBFunc>
GRACE_HOST void morton_keys(
    PrimitiveIter d_prims_iter,
    const size_t N_primitives,
    const Real3 AABB_top,
    const Real3 AABB_bot,
    KeyIter d_keys_iter,
    const AABBFunc AABB)
{
    typedef typename std::iterator_traits<KeyIter>::value_type KeyType;

    const int MAX_KEY_63 = (1u << 21) - 1;
    const int MAX_KEY_30 = (1u << 10) - 1;
    const int span = CHAR_BIT * sizeof(KeyType) > 32 ? MAX_KEY_63 : MAX_KEY_30;
    float3 scale = make_float3(span / (AABB_top.x - AABB_bot.x),
                               span / (AABB_top.y - AABB_bot.y),
                               span / (AABB_top.z - AABB_bot.z));

    morton::morton_keys(d_prims_iter, N_primitives, scale, d_keys_iter, AABB);
}

template <typename TPrimitive, typename Real3, typename KeyType,
          typename AABBFunc>
GRACE_HOST void morton_keys(
    const thrust::device_vector<TPrimitive>& d_primitives,
    const Real3 AABB_top,
    const Real3 AABB_bot,
    thrust::device_vector<KeyType>& d_keys,
    const AABBFunc AABB)
{
    const size_t N_primitives = d_primitives.size();
    const TPrimitive* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    KeyType* keys_ptr = thrust::raw_pointer_cast(d_keys.data());

    morton_keys(prims_ptr, d_primitives.size(), AABB_top, AABB_bot, keys_ptr,
                AABB);
}

// Wrappers to compute the AABB containing all primitives.
// Requires O(N_primitives) temporary storage.
template <typename PrimitiveIter, typename KeyIter, typename AABBFunc>
GRACE_HOST void morton_keys(
    PrimitiveIter d_prims_iter,
    const size_t N_primitives,
    KeyIter d_keys_iter,
    const AABBFunc AABB)
{
    // grace::min_max_{x,y,z} use Thrust functions, which must accept either
    // Thrust iterators or device pointers.
    // We therefore cannot create a d_centroids_iter which converts primitives
    // to centroids in-place, because such an iterator would need to be
    // dereferenceable on both the host and the device. The underlying thrust
    // kernel would dereference on the device, but grace::min_max_x would
    // dereference the result on the host. This will work if PrimitiveIter is
    // actually a Thrust iterator, but not otherwise!

    thrust::device_vector<float3> d_centroids(N_primitives);
    float3* d_centroids_ptr = thrust::raw_pointer_cast(d_centroids.data());

    AABB::compute_centroids(d_prims_iter, N_primitives, d_centroids_ptr, AABB);

    float3 mins, maxs;
    min_vec3(d_centroids_ptr, N_primitives, &mins);
    max_vec3(d_centroids_ptr, N_primitives, &maxs);

    morton_keys(d_prims_iter, N_primitives, maxs, mins, d_keys_iter, AABB);
}

template <typename TPrimitive, typename KeyType, typename AABBFunc>
GRACE_HOST void morton_keys(
    const thrust::device_vector<TPrimitive>& d_primitives,
    thrust::device_vector<KeyType>& d_keys,
    const AABBFunc AABB)
{
    const TPrimitive* d_prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    KeyType* d_keys_ptr = thrust::raw_pointer_cast(d_keys.data());

    morton_keys(d_prims_ptr, d_primitives.size(), d_keys_ptr, AABB);
}

} // namespace grace
