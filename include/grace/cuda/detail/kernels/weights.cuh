#pragma once

#include "grace/cuda/error.cuh"
#include "grace/config.h"

#include <thrust/device_vector.h>

namespace grace {

namespace detail {

template <typename Real, typename IdxType>
__global__ void multiply_by_weights_kernel(
    const Real* unweighted,
    size_t N_unweighted,
    const Real* weights,
    const IdxType* weight_map,
    Real* weighted)
{
    for(unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        tid < N_unweighted;
        tid += blockDim.x * gridDim.x)
    {
        IdxType weight_index = weight_map[tid];
        weighted[tid] = weights[weight_index] * unweighted[tid];
    }
}

template <typename Real, typename IdxType>
GRACE_HOST void multiply_by_weights(
    const Real* unweighted,
    const size_t N_unweighted,
    const Real* weights,
    const IdxType* weight_map,
    Real* weighted)
{
    detail::multiply_by_weights_kernel<<<48, 512>>>(
        unweighted, N_unweighted, weights, weight_map, weighted
    );
    GRACE_CUDA_KERNEL_CHECK();
}

template <typename Real, typename IdxType>
GRACE_HOST void multiply_by_weights(
    const thrust::device_vector<Real>& d_unweighted,
    const thrust::device_vector<Real>& d_weights,
    const thrust::device_vector<IdxType>& d_weight_map,
    thrust::device_vector<Real>& d_weighted)
{
    multiply_by_weights(
        thrust::raw_pointer_cast(d_unweighted.data()),
        d_unweighted.size(),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_weight_map.data()),
        thrust::raw_pointer_cast(d_weighted.data())
    );
}

} // namespace detail

} // namespace grace
