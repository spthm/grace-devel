#pragma once

#include "grace/cuda/detail/kernel_config.h"

#include "grace/generic/functors/aabb.h"

#include "grace/error.h"
#include "grace/types.h"
#include "grace/vector.h"

#include <algorithm>
#include <iterator>

namespace grace {

namespace AABB {

template <typename PrimitiveIter, typename T, typename CentroidFunc>
__global__ void compute_centroids_kernel(
    PrimitiveIter primitives,
    const size_t N_primitives,
    Vector<3, T>* centroids,
    const CentroidFunc centroid)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_primitives)
    {
        TPrimitive prim = primitives[tid];
        centroids[tid] = centroid(prim);

        tid += blockDim.x * gridDim.x;
    }
}

template <typename PrimitiveIter, typename T, typename CentroidFunc>
GRACE_HOST void compute_centroids(
    PrimitiveIter d_prims_iter,
    const size_t N_primitives,
    Vector<3, T>* d_centroid_ptr,
    const CentroidFunc centroid)
{
    const int NT = 256;
    int blocks = std::min(MAX_BLOCKS, (int) ((N_primitives + NT - 1) / NT));
    compute_centroids_kernel<<<blocks, NT>>>(
        d_prims_iter,
        N_primitives,
        d_centroid_ptr,
        centroid);
    GRACE_KERNEL_CHECK();
}

} // namespace AABB

} // namespace grace
