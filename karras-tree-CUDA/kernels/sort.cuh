#pragma once

#include <thrust/device_vector>
#include <thrust/sort.h>

namespace grace {

template <typename UInteger, typename Float4, typename Float>
void sort_by_key(thrust::device_vector<UInteger>& d_keys,
                 thrust::device_vector<Float4>& d_spheres,
                 thrust::device_vector<Float>& d_property)
{
    thrust::device_vector<UInteger> d_keys2 = d_keys;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());

    thrust::sort_by_key(d_keys2.begin(), d_keys2.end(), d_property.begin());
}

} // namespace grace
