#pragma once

#include "../../moderngpu/include/kernels/segmentedsort.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace grace {

void offsets_to_segments(const thrust::device_vector<unsigned int>& d_offsets,
                         thrust::device_vector<unsigned int>& d_segments)
{
    size_t N_offsets = d_offsets.size();
    thrust::constant_iterator<unsigned int> first(1);
    thrust::constant_iterator<unsigned int> last = first + N_offsets;

    // Suppose offsets = [0, 3, 3, 7]
    // scatter value 1 at offsets[1:N] into segments:
    //    => ray_segments = [0, 0, 0, 1, 0, 0, 0, 1(, 0 ... )]
    // inclusive_scan:
    //    => ray_segments = [0, 0, 0, 1, 1, 1, 1, 2(, 2 ... )]
    // Note that we do not scatter a 1 into ray_segments[offsets[0]].
    thrust::scatter(first+1, last,
                    d_offsets.begin()+1,
                    d_segments.begin());
    thrust::inclusive_scan(d_segments.begin(), d_segments.end(),
                           d_segments.begin());
}

template <typename UInteger, typename T>
void order_by_index(const thrust::device_vector<UInteger>& d_indices,
                    thrust::device_vector<T>& d_unordered)
{
    thrust::device_vector<T> d_tmp = d_unordered;
    thrust::gather(d_indices.begin(), d_indices.end(),
                   d_tmp.begin(),
                   d_unordered.begin());
}

template <typename T, typename UInteger>
void sort_and_map(thrust::device_vector<T>& d_unsorted,
                  thrust::device_vector<UInteger>& d_map)
{
    thrust::sequence(d_map.begin(), d_map.end(), 0u);
    thrust::sort_by_key(d_unsorted.begin(), d_unsorted.end(), d_map.begin());
}

// Like sort_and_map, but does not touch the original, unsorted vector.
template <typename T, typename UInteger>
void sort_map(thrust::device_vector<T>& d_unsorted,
              thrust::device_vector<UInteger>& d_map)
{
    thrust::sequence(d_map.begin(), d_map.end(), 0u);
    thrust::device_vector<T> d_tmp = d_unsorted;
    thrust::sort_by_key(d_tmp.begin(), d_tmp.end(), d_map.begin());
}

template <typename T_key, typename Ta, typename Tb>
void sort_by_key(thrust::host_vector<T_key>& h_keys,
                 thrust::host_vector<Ta>& h_a,
                 thrust::host_vector<Tb>& h_b)
{
    thrust::host_vector<T_key> h_keys2 = h_keys;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_a.begin());

    thrust::sort_by_key(h_keys2.begin(), h_keys2.end(), h_b.begin());
}

template <typename T_key, typename Ta, typename Tb>
void sort_by_key(thrust::device_vector<T_key>& d_keys,
                 thrust::device_vector<Ta>& d_a,
                 thrust::device_vector<Tb>& d_b)
{
    thrust::device_vector<T_key> d_keys2 = d_keys;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_a.begin());

    thrust::sort_by_key(d_keys2.begin(), d_keys2.end(), d_b.begin());
}

template <typename Float, typename T>
void sort_by_distance(thrust::device_vector<Float>& d_hit_distances,
                      const thrust::device_vector<int>& d_ray_offsets,
                      thrust::device_vector<unsigned int>& d_hit_indices,
                      thrust::device_vector<T>& d_hit_data)
{
    // MGPU calls require a context.
    int device_ID = 0;
    cudaGetDevice(&device_ID);
    mgpu::ContextPtr mgpu_context_ptr = mgpu::CreateCudaDevice(device_ID);

    // d_indices will be a map to reorder the input data.
    thrust::device_vector<unsigned int> d_indices(d_hit_distances.size());
    thrust::sequence(d_indices.begin(), d_indices.end(), 0u);

    // First, sort the hit distances and the indicies within the segments
    // defined by d_ray_offsets, i.e. sort each ray and its indices by distance.
    // The distances are the keys, and the ordered indices are the values.
    mgpu::SegSortPairsFromIndices(
        thrust::raw_pointer_cast(d_hit_distances.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        d_hit_distances.size(),
        thrust::raw_pointer_cast(d_ray_offsets.data()),
        d_ray_offsets.size(),
        *mgpu_context_ptr);
    // Second, reorder the hit indices and hit data by the map produced in the
    // above sort.
    order_by_index(d_indices, d_hit_indices);
    order_by_index(d_indices, d_hit_data);
}

} // namespace grace
