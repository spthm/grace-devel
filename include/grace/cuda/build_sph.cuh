#pragma once

#include "grace/cuda/nodes.h"

#include "grace/types.h"

#include <thrust/device_vector.h>

namespace grace {

// Real4 should be float4 or double4.
// KeyType should be grace::uinteger{32,64}.
template <typename Real4, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<KeyType>& d_keys);

template <typename Real3, typename Real4, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Real4>& d_spheres,
    const Real3 bot,
    const Real3 top,
    thrust::device_vector<KeyType>& d_keys);

// Real4 should be float4 or double4.
// Generates 30-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename Real4>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Real4>& d_spheres);

template <typename Real3, typename Real4>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Real4>& d_spheres,
    const Real3 bot,
    const Real3 top);

// Real4 should be float4 or double4.
// Generates 63-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename Real4>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Real4>& d_spheres);

template <typename Real3, typename Real4>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Real4>& d_spheres,
    const Real3 bot,
    const Real3 top);

// Real4 should be float4 or double4.
// Real must be the float or double, respectively.
template <typename Real4, typename Real>
GRACE_HOST void euclidean_deltas_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<Real>& d_deltas);

// Real4 should be float4 or double4.
// Real must be the float or double, respectively.
template <typename Real4, typename Real>
GRACE_HOST void surface_area_deltas_sph(
    const thrust::device_vector<Real4>& d_spheres,
    thrust::device_vector<Real>& d_deltas);

// KeyType should be grace::uinteger{32,64}.
// DeltaType should be grace::uinteger{32, 64}.
template <typename KeyType, typename DeltaType>
GRACE_HOST void XOR_deltas_sph(
    const thrust::device_vector<KeyType>& d_morton_keys,
    thrust::device_vector<DeltaType>& d_deltas);

// Real4 should be float4 or double4.
template <typename Real4, typename DeltaType>
GRACE_HOST void ALBVH_sph(
    const thrust::device_vector<Real4>& d_spheres,
    const thrust::device_vector<DeltaType>& d_deltas,
    Tree& d_tree);

} // namespace grace

#include "grace/cuda/detail/build_sph-inl.cuh"
