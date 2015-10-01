#pragma once

#include "../device/aabb.cuh"

#include "../kernel_config.h"

#include <iterator>

namespace grace {

namespace AABB {

// Centroid
template <typename PrimitiveIter, typename AABBFunc>
__global__ void compute_centroids_kernel(
    PrimitiveIter primitives,
    const size_t N_primitives,
    float3* centroids,
    const AABBFunc AABB)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_primitives)
    {
        TPrimitive prim = primitives[tid];
        float3 bot, top;
        AABB(prim, &bot, &top);
        centroids[tid] = AABB_centroid(bot, top);
    }
}

template <typename PrimitiveIter, typename CentroidIter, typename AABBFunc>
GRACE_HOST void compute_centroids(
    PrimitiveIter d_prims_iter,
    const size_t N_primitives,
    CentroidIter d_centroid_iter,
    const AABBFunc AABB)
{
    int blocks = min(MAX_BLOCKS, (int) ((N_primitives + 256 - 1) / 256));
    compute_centroids_kernel(d_prims_iter, N_primitives, d_centroid_iter, AABB);
}

} // namespace AABB

} // namespace grace
