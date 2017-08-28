#pragma once

#include "grace/cuda/bvh.cuh"

#include "grace/ray.h"
#include "grace/sphere.h"
#include "grace/types.h"

#include <thrust/device_vector.h>

namespace grace {

//-----------------------------------------------------------------------------
// C-like convenience wrappers for common forms of the tracing kernel.
//-----------------------------------------------------------------------------

// T defines the internal precision of the intersection test.
// This is an API choice. The underlying machinery support different precisions
// for the Sphere<T> and other computations.
template <typename T>
GRACE_HOST void trace_hitcounts_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<int>& d_hit_counts);

// T defines the internal precision of the intersection test, and of
// intermediate computations for the output(s).
// This is an API choice. The underlying machinery support different precisions
// for the Sphere<T> and other computations.
template <typename T, typename OutType>
GRACE_HOST void trace_cumulative_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    thrust::device_vector<OutType>& d_cumulated);

// T defines the internal precision of the intersection test, and of
// intermediate computations for the output(s).
// This is an API choice. The underlying machinery support different precisions
// for the Sphere<T> and other computations.
template <typename T, typename IndexType, typename OutType>
GRACE_HOST void trace_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require ray offsets to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    thrust::device_vector<OutType>& d_hit_integrals,
    thrust::device_vector<OutType>& d_hit_distances);

// T defines the internal precision of the intersection test, and of
// intermediate computations for the output(s).
// This is an API choice. The underlying machinery support different precisions
// for the Sphere<T> and other computations.
template <typename T, typename IndexType, typename OutType>
GRACE_HOST void trace_with_sentinels_sph(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Tree& d_tree,
    // SGPU's segmented scans and sorts require this to be int.
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<IndexType>& d_hit_indices,
    const int index_sentinel,
    thrust::device_vector<OutType>& d_hit_integrals,
    const OutType integral_sentinel,
    thrust::device_vector<OutType>& d_hit_distances,
    const OutType distance_sentinel);

} // namespace grace

#include "grace/cuda/detail/trace_sph-inl.cuh"
