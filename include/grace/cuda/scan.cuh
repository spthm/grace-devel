#pragma once

#include "grace/types.h"

#include "grace/cuda/kernels/weights.cuh"

#include "grace/external/sgpu/kernels/segscancsr.cuh"

#include <thrust/device_vector.h>

namespace grace {

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
    detail::multiply_by_weights(d_to_sum, d_weights, d_weight_map, d_weighted);

    // Segmented, cumulative sum such that d_sum[i] = cumulative sum *up to*
    // (not including, i.e. an exclusive sum) the ith weighted value.
    grace::exclusive_segmented_scan(d_segment_offsets, d_weighted, d_sum);
}

} // namespace grace
