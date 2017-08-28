#pragma once

#include "grace/config.h"

#include <thrust/device_vector.h>

namespace grace {

// d_data and d_results may be the same vector.
template <typename Real>
GRACE_HOST void exclusive_segmented_scan(
    const thrust::device_vector<int>& d_segment_offsets,
    thrust::device_vector<Real>& d_data,
    thrust::device_vector<Real>& d_results);

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
    thrust::device_vector<Real>& d_sum);

} // namespace grace

#include "grace/cuda/detail/scan-inl.cuh"
