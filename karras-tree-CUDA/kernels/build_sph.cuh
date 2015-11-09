#pragma once

#include "albvh.cuh"
#include "morton.cuh"
#include "../device/build_functors.cuh"
#include "../types.h"

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
    morton_keys(d_spheres, d_keys, AABB_sphere());
}

template <typename Real4, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Real4>& d_spheres,
    const float3 top,
    const float3 bot,
    thrust::device_vector<KeyType>& d_keys)
{
    morton_keys(d_spheres, top, bot, d_keys, AABB_sphere());
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

template <typename Real4>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Real4>& d_spheres,
    const float3 top,
    const float3 bot)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, top, bot, d_keys);
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

template <typename Real4>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Real4>& d_spheres,
    const float3 top,
    const float3 bot)
{
    thrust::device_vector<grace::uinteger64> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, top, bot, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}


// Real4 should be float4 or double4.
// Real must be the float or double, respectively.
template <typename Real4, typename Real>
GRACE_HOST void euclidean_deltas_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<Real>& d_deltas)
{
    compute_deltas(d_spheres, d_deltas, Delta_sphere_euclidean());
}

// Real4 should be float4 or double4.
// Real must be the float or double, respectively.
template <typename Real4, typename Real>
GRACE_HOST void surface_area_deltas_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<Real>& d_deltas)
{
    compute_deltas(d_spheres, d_deltas, Delta_sphere_SA());
}

// KeyType should be grace::uinteger{32,64}.
// DeltaType should be grace::uinteger{32, 64}.
template <typename KeyType, typename DeltaType>
GRACE_HOST void XOR_deltas_sph(
    const thrust::device_vector<KeyType>& d_morton_keys,
    thrust::device_vector<DeltaType>& d_deltas)
{
    compute_deltas(d_morton_keys, d_deltas, Delta_XOR());
}

// Real4 should be float4 or double4.
template <typename Real4, typename DeltaType>
GRACE_HOST void ALBVH_sph(
    const thrust::device_vector<Real4>& d_spheres,
    const thrust::device_vector<DeltaType>& d_deltas,
    Tree& d_tree)
{
    build_ALBVH(d_tree, d_spheres, d_deltas, AABB_sphere());
}

} // namespace grace
