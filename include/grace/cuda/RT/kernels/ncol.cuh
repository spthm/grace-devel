#pragma once

#include "grace/error.h"
#include "grace/types.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace grace {

namespace gpu {

template <typename Float>
__global__ void multiply_by_weights(const Float* unweighted,
                                    const Float* weights,
                                    const unsigned int* weight_map,
                                    size_t N_unweighted,
                                    Float* weighted)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int weight_index;

    while (tid < N_unweighted)
    {
        weight_index = weight_map[tid];
        weighted[tid] = weights[weight_index] * unweighted[tid];
        tid += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

template <typename Float>
GRACE_HOST void cum_weighted_sum(
    const thrust::device_vector<Float>& d_to_sum,
    const thrust::device_vector<Float>& d_weights,
    const thrust::device_vector<unsigned int>& d_weight_map,
    const thrust::device_vector<int>& d_segment_offsets,
    thrust::device_vector<Float>& d_sum)
{
    // Initialize d_sum with the weighted inputs
    // (not cumulative!) such that
    //     weighted[i] = weights[weight_map[i]] * to_sum[i].
    // e.g.
    //     Ncol[i] = Ncol_factor[particle_indices[i]] * kernel_integrals[i]
    thrust::device_vector<Float> d_weighted(d_to_sum.size());
    gpu::multiply_by_weights<<<48, 512>>>(
        thrust::raw_pointer_cast(d_to_sum.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_weight_map.data()),
        d_weighted.size(),
        thrust::raw_pointer_cast(d_weighted.data()));
    GRACE_KERNEL_CHECK();

    // Segmented, cumulative sum such that d_sum[i] = cumulative sum *up to*
    // (not including, i.e. an exclusive sum) the ith weighted value.
    // E.g. Ncols[i] = cumulative Ncol *up to* (not including) the ith
    // intersected particle, for each ray marked by the segments.
    grace::exclusive_segmented_scan(d_segment_offsets, d_weighted, d_sum);
}

} // namespace grace
