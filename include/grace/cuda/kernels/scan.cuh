#pragma once

#include "grace/error.h"
#include "grace/types.h"

#include "grace/external/sgpu/kernels/segscancsr.cuh"

#include <thrust/device_vector.h>

namespace grace {

namespace gpu {

template <typename Real>
__global__ void multiply_by_weights(const Real* unweighted,
                                    const Real* weights,
                                    const unsigned int* weight_map,
                                    size_t N_unweighted,
                                    Real* weighted)
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

// d_data and d_results may be the same vector.
template <typename Real>
GRACE_HOST void exclusive_segmented_scan(
    const thrust::device_vector<int>& d_segment_offsets,
    thrust::device_vector<Real>& d_data,
    thrust::device_vector<Real>& d_results)
{
    // SGPU calls require a context.
    int device_ID = 0;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_ID));
    sgpu::ContextPtr sgpu_context_ptr = sgpu::CreateCudaDevice(device_ID);

    std::auto_ptr<sgpu::SegCsrPreprocessData> pp_data_ptr;

    size_t N_data = d_data.size();
    size_t N_segments = d_segment_offsets.size();

    sgpu::SegScanCsrPreprocess<Real>(
        N_data, thrust::raw_pointer_cast(d_segment_offsets.data()),
        N_segments, true, &pp_data_ptr, *sgpu_context_ptr);

    sgpu::SegScanApply<sgpu::SgpuScanTypeExc>(*pp_data_ptr,
        thrust::raw_pointer_cast(d_data.data()), Real(0), sgpu::plus<Real>(),
        thrust::raw_pointer_cast(d_results.data()), *sgpu_context_ptr);
}

// d_weight_map should behave such that the weight for a value at index i in
// d_to_sum is located at index j = d_weight_map[i] in d_weights.
// That is,
//     weighted_values[i] = d_to_sum[i] * d_weights[d_weight_map[i]]
template <typename Real>
GRACE_HOST void weighted_exclusive_segmented_scan(
    const thrust::device_vector<Real>& d_to_sum,
    const thrust::device_vector<Real>& d_weights,
    const thrust::device_vector<unsigned int>& d_weight_map,
    const thrust::device_vector<int>& d_segment_offsets,
    thrust::device_vector<Real>& d_sum)
{
    // Initialize the weighted inputs, d_weighted.
    thrust::device_vector<Real> d_weighted(d_to_sum.size());
    gpu::multiply_by_weights<<<48, 512>>>(
        thrust::raw_pointer_cast(d_to_sum.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_weight_map.data()),
        d_weighted.size(),
        thrust::raw_pointer_cast(d_weighted.data()));
    GRACE_KERNEL_CHECK();

    // Segmented, cumulative sum such that d_sum[i] = cumulative sum *up to*
    // (not including, i.e. an exclusive sum) the ith weighted value.
    grace::exclusive_segmented_scan(d_segment_offsets, d_weighted, d_sum);
}

} // namespace grace
