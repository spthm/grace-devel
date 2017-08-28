#pragma once

#include "grace/config.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace grace {

GRACE_HOST void offsets_to_segments(
    const thrust::device_vector<int>& d_offsets,
    thrust::device_vector<int>& d_segments);

template <typename IndexType, typename T>
GRACE_HOST void order_by_index(
    const thrust::device_vector<IndexType>& d_indices,
    thrust::device_vector<T>& d_unordered);

template <typename T, typename IndexType>
GRACE_HOST void sort_and_map(
    thrust::device_vector<T>& d_unsorted,
    thrust::device_vector<IndexType>& d_map);

// Like sort_and_map, but does not touch the original, unsorted vector.
template <typename T, typename IndexType>
GRACE_HOST void sort_map(
    thrust::device_vector<T>& d_unsorted,
    thrust::device_vector<IndexType>& d_map);

template <typename T_key, typename Ta, typename Tb>
GRACE_HOST void sort_by_key(
    thrust::host_vector<T_key>& h_keys,
    thrust::host_vector<Ta>& h_a,
    thrust::host_vector<Tb>& h_b);

template <typename T_key, typename Ta, typename Tb>
GRACE_HOST void sort_by_key(
    thrust::device_vector<T_key>& d_keys,
    thrust::device_vector<Ta>& d_a,
    thrust::device_vector<Tb>& d_b);

template <typename Real, typename IndexType, typename T>
GRACE_HOST void sort_by_distance(
    thrust::device_vector<Real>& d_hit_distances,
    const thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    thrust::device_vector<T>& d_hit_data);

} // namespace grace

#include "grace/cuda/detail/sort-inl.cuh"
