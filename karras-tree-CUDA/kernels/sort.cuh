#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

namespace grace {

template <typename UInteger, typename Ta, typename Tb>
void sort_by_key(thrust::device_vector<UInteger>& d_keys,
                 thrust::device_vector<Ta>& d_a,
                 thrust::device_vector<Tb>& d_b)
{
    thrust::device_vector<UInteger> d_keys2 = d_keys;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_a.begin());

    thrust::sort_by_key(d_keys2.begin(), d_keys2.end(), d_b.begin());
}

template <typename UInteger, typename Ta, typename Tb>
void sort_by_key(thrust::host_vector<UInteger>& h_keys,
                 thrust::host_vector<Ta>& h_a,
                 thrust::host_vector<Tb>& h_b)
{
    thrust::host_vector<UInteger> h_keys2 = h_keys;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_a.begin());

    thrust::sort_by_key(h_keys2.begin(), h_keys2.end(), h_b.begin());
}

} // namespace grace
