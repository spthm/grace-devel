#pragma once

#include "grace/cuda/nodes.h"

#include "grace/sphere.h"
#include "grace/types.h"

#include <thrust/device_vector.h>

namespace grace {

// KeyType should be grace::uinteger{32,64}.
template <typename T, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    thrust::device_vector<KeyType>& d_keys);

template <typename T, typename Real3, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Real3 bot,
    const Real3 top,
    thrust::device_vector<KeyType>& d_keys);

// Generates 30-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename T>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres);

template <typename T, typename Real3>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres,
    const Real3 bot,
    const Real3 top);

// Generates 63-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename T>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres);

template <typename T, typename Real3>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres,
    const Real3 bot,
    const Real3 top);

template <typename T>
GRACE_HOST void euclidean_deltas_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    thrust::device_vector<float>& d_deltas);

template <typename T>
GRACE_HOST void surface_area_deltas_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    thrust::device_vector<float>& d_deltas);

// KeyType should be grace::uinteger{32,64}.
// Note that the resulting deltas have the same type as the key for this
// distance metric.
template <typename KeyType>
GRACE_HOST void XOR_deltas_sph(
    const thrust::device_vector<KeyType>& d_morton_keys,
    thrust::device_vector<KeyType>& d_deltas);

template <typename T, typename DeltaType>
GRACE_HOST void ALBVH_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const thrust::device_vector<DeltaType>& d_deltas,
    Tree& d_tree);

} // namespace grace

#include "grace/cuda/detail/build_sph-inl.cuh"
