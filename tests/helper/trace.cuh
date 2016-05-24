#pragma once

#include "grace/cuda/nodes.h"
#include "grace/cuda/sort.cuh"
#include "grace/cuda/trace_sph.cuh"
#include "grace/ray.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Returns host vectors containing ray-particle intersection distances and
// per-ray offsets into that vector.
// Real4 and Real must be same precision.
template <typename Real, typename Real4>
void trace_distances(const thrust::device_vector<grace::Ray>& d_rays,
                     const thrust::device_vector<Real4>& d_spheres,
                     const grace::Tree& d_tree,
                     thrust::host_vector<int>& h_offsets,
                     thrust::host_vector<Real>& h_distances)
{
    thrust::device_vector<int> d_offsets(d_rays.size());
    // Below will be resized.
    thrust::device_vector<Real> d_distances;
    // Below required but not returned.
    thrust::device_vector<Real> d_integrals;
    thrust::device_vector<int> d_indices;

    grace::trace_sph(d_rays,
                     d_spheres,
                     d_tree,
                     d_offsets,
                     d_indices,
                     d_integrals,
                     d_distances);
    h_offsets = d_offsets;

    grace::sort_by_distance(d_distances,
                            d_offsets,
                            d_indices,
                            d_integrals);

    h_distances = d_distances;
}
