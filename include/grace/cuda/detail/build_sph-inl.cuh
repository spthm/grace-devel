#pragma once

#include "grace/cuda/detail/kernels/albvh.cuh"
#include "grace/cuda/detail/kernels/morton.cuh"

#include "grace/cuda/nodes.h"

#include "grace/generic/functors/albvh.h"
#include "grace/generic/functors/centroid.h"

#include "grace/types.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace grace {

// Real4 should be float4 or double4.
// KeyType should be grace::uinteger{32,64}.
template <typename Real4, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<KeyType>& d_keys)
{
    morton_keys(d_spheres, d_keys, CentroidSphere());
}

template <typename Real3, typename Real4, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Real4>& d_spheres,
    const Real3 bot,
    const Real3 top,
    thrust::device_vector<KeyType>& d_keys)
{
    morton_keys(d_spheres, bot, top, d_keys, CentroidSphere());
}

// Real4 should be float4 or double4.
// Generates 30-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename Real4>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Real4>& d_spheres)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

template <typename Real3, typename Real4>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Real4>& d_spheres,
    const Real3 bot,
    const Real3 top)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, bot, top, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

// Real4 should be float4 or double4.
// Generates 63-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename Real4>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Real4>& d_spheres)
{
    thrust::device_vector<grace::uinteger64> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

template <typename Real3, typename Real4>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Real4>& d_spheres,
    const Real3 bot,
    const Real3 top)
{
    thrust::device_vector<grace::uinteger64> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, bot, top, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

// Real4 should be float4 or double4.
// Real must be the float or double, respectively.
template <typename Real4, typename Real>
GRACE_HOST void euclidean_deltas_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<Real>& d_deltas)
{
    compute_deltas(d_spheres, d_deltas,
                   DeltaEuclidean<const Real4*, CentroidSphere>());
}

// Real4 should be float4 or double4.
// Real must be the float or double, respectively.
template <typename Real4, typename Real>
GRACE_HOST void surface_area_deltas_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<Real>& d_deltas)
{
    compute_deltas(d_spheres, d_deltas,
                   DeltaSurfaceArea<const Real4*, AABBSphere>());
}

// KeyType should be grace::uinteger{32,64}.
// DeltaType should be grace::uinteger{32, 64}.
template <typename KeyType, typename DeltaType>
GRACE_HOST void XOR_deltas_sph(
    const thrust::device_vector<KeyType>& d_morton_keys,
    thrust::device_vector<DeltaType>& d_deltas)
{
    compute_deltas(d_morton_keys, d_deltas, DeltaXOR());
}

// Real4 should be float4 or double4.
template <typename Real4, typename DeltaType>
GRACE_HOST void ALBVH_sph(
    const thrust::device_vector<Real4>& d_spheres,
    const thrust::device_vector<DeltaType>& d_deltas,
    Tree& d_tree)
{
    build_ALBVH(d_tree, d_spheres, d_deltas, AABBSphere());
}

} // namespace grace
