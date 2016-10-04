#pragma once

#include "triangle.cuh"

#include "grace/cuda/kernels/albvh.cuh"
#include "grace/cuda/kernels/morton.cuh"

#include "grace/generic/functors/albvh.h"

#include "grace/types.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

// KeyType should be grace::uinteger{32,64}.
template <typename KeyType>
GRACE_HOST void morton_keys_tri(
    const thrust::device_vector<Triangle>& d_tris,
    thrust::device_vector<KeyType>& d_keys)
{
    grace::morton_keys(d_tris, d_keys, TriangleCentroid());
}

template <typename Real3, typename KeyType>
GRACE_HOST void morton_keys_tri(
    const thrust::device_vector<Triangle>& d_tris,
    const Real3 bot,
    const Real3 top,
    thrust::device_vector<KeyType>& d_keys)
{
    grace::morton_keys(d_tris, bot, top, d_keys, TriangleCentroid());
}

GRACE_HOST void build_tree_tris(
    thrust::device_vector<Triangle>& d_tris,
    grace::Tree& d_tree)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_tris.size());
    thrust::device_vector<grace::uinteger32> d_deltas(d_tris.size() + 1);

    morton_keys_tri(d_tris, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_tris.begin());
    grace::compute_deltas(d_keys, d_deltas, grace::DeltaXOR());
    grace::build_ALBVH(d_tree, d_tris, d_deltas, TriangleAABB());
}
