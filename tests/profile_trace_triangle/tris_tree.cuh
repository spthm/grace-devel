#pragma once

#include "triangle.cuh"

// #include "grace/cuda/kernels/albvh.cuh"
#include "grace/cuda/kernels/morton.cuh"

// #include "grace/generic/functors/albvh.h"

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

// Generates 30-bit Morton keys.
// Sorts triangles by the Morton keys.
// Requires O(N) on-device temporary storage.
GRACE_HOST void morton_keys30_sort_tri(
    thrust::device_vector<Triangle>& d_tris)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_tris.size());
    morton_keys_tri(d_tris, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_tris.begin());
}

template <typename Real3>
GRACE_HOST void morton_keys30_sort_tri(
    thrust::device_vector<Triangle>& d_tris,
    const Real3 bot,
    const Real3 top)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_tris.size());
    morton_keys_tri(d_tris, bot, top, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_tris.begin());
}

// // Real must be the float or double, respectively.
// template <typename Real>
// GRACE_HOST void euclidean_deltas_tri(
//     const thrust::device_vector<Triangle>& d_tris,
//     thrust::device_vector<Real>& d_deltas)
// {
//     compute_deltas(d_tris, d_deltas,
//                    DeltaEuclidean<const Real4*, TriangleCentroid>());
// }

// // Real must be the float or double, respectively.
// template <typename Real>
// GRACE_HOST void surface_area_deltas_tri(
//     const thrust::device_vector<Triangle>& d_tris,
//     thrust::device_vector<Real>& d_deltas)
// {
//     compute_deltas(d_tris, d_deltas,
//                    DeltaSurfaceArea<const Real4*, TriangleAABB>());
// }

// // KeyType should be grace::uinteger{32,64}.
// // DeltaType should be grace::uinteger{32, 64}.
// template <typename KeyType, typename DeltaType>
// GRACE_HOST void XOR_deltas_tri(
//     const thrust::device_vector<KeyType>& d_morton_keys,
//     thrust::device_vector<DeltaType>& d_deltas)
// {
//     compute_deltas(d_morton_keys, d_deltas, DeltaXOR());
// }

// // Real4 should be float4 or double4.
// template <typename DeltaType, typename AABBTriangle>
// GRACE_HOST void ALBVH_tri(
//     const thrust::device_vector<Triangle>& d_tris,
//     const thrust::device_vector<DeltaType>& d_deltas,
//     Tree& d_tree)
// {
//     build_ALBVH(d_tree, d_tris, d_deltas, aabb_tri);
// }

GRACE_HOST void build_tree_tris(
    thrust::device_vector<Triangle>& d_tris,
    grace::Tree& d_tree)
{
    thrust::device_vector<grace::uinteger32> d_deltas(d_tris.size() + 1);

    morton_keys30_sort_tri(d_tris);
    grace::compute_deltas(d_tris, d_deltas, DeltaXOR());
    grace::build_ALBVH(d_tree, d_tris, d_deltas, TriangleAABB());
}
