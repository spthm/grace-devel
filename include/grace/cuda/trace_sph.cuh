#pragma once

#include "grace/cuda/nodes.h"

#include "grace/ray.h"
#include "grace/types.h"

#include <thrust/device_vector.h>

namespace grace {

//-----------------------------------------------------------------------------
// C-like convenience wrappers for common forms of the tracing kernel.
//-----------------------------------------------------------------------------

template <typename Real4>
GRACE_HOST void trace_hitcounts_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<int>& d_hit_counts);

template <typename Real4, typename Real>
GRACE_HOST void trace_cumulative_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<Real>& d_cumulated);

template <typename Real4, typename IndexType, typename Real>
GRACE_HOST void trace_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require ray offsets to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    thrust::device_vector<Real>& d_hit_integrals,
    thrust::device_vector<Real>& d_hit_distances);

template <typename Real4, typename IndexType, typename Real>
GRACE_HOST void trace_with_sentinels_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Real4>& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require this to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    const int index_sentinel,
    thrust::device_vector<Real>& d_hit_integrals,
    const Real integral_sentinel,
    thrust::device_vector<Real>& d_hit_distances,
    const Real distance_sentinel);

} // namespace grace

#include "grace/cuda/detail/trace_sph-inl.cuh"
