#pragma once

#include "triangle.cuh"

#include "grace/cuda/detail/kernels/albvh.cuh"
#include "grace/cuda/detail/kernels/morton.cuh"

#include "grace/cuda/nodes.h"

#include "grace/generic/functors/albvh.h"

#include "grace/aabb.h"
#include "grace/types.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

// If not NULL, aabb will be populated with the minimum and maximum triangle
// centroid bounds.
inline void build_tree_tris(
    thrust::device_vector<Triangle>& d_tris,
    grace::Tree& d_tree,
    grace::AABB<float>* aabb = NULL)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_tris.size());
    thrust::device_vector<grace::uinteger32> d_deltas(d_tris.size() + 1);

    grace::morton_keys(d_tris, d_keys, TriangleCentroid(), aabb);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_tris.begin());
    grace::compute_deltas(d_keys, d_deltas, grace::DeltaXOR());
    grace::build_ALBVH(d_tree, d_tris, d_deltas, TriangleAABB());
}
