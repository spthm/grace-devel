#pragma once

#include "grace/cuda/detail/kernels/albvh.cuh"
#include "grace/cuda/detail/kernels/morton.cuh"

#include "grace/cuda/nodes.h"

#include "grace/generic/functors/albvh.h"
#include "grace/generic/functors/centroid.h"

#include "grace/sphere.h"
#include "grace/types.h"
#include "grace/vector.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace grace {

// KeyType should be grace::uinteger{32,64}.
template <typename T, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    thrust::device_vector<KeyType>& d_keys)
{
    morton_keys(d_spheres, d_keys, CentroidSphere<T>());
}

template <typename T, typename KeyType>
GRACE_HOST void morton_keys_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const Vector<3, T>& bot,
    const Vector<3, T>& top,
    thrust::device_vector<KeyType>& d_keys)
{
    morton_keys(d_spheres, bot, top, d_keys, CentroidSphere<T>());
}

// Generates 30-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename T>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

template <typename T>
GRACE_HOST void morton_keys30_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres,
    const Vector<3, T>& bot,
    const Vector<3, T>& top)
{
    thrust::device_vector<grace::uinteger32> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, bot, top, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

// Generates 63-bit Morton keys.
// Sorts spheres by the Morton keys.
// Requires O(N) on-device temporary storage.
template <typename T>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres)
{
    thrust::device_vector<grace::uinteger64> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

template <typename T>
GRACE_HOST void morton_keys63_sort_sph(
    thrust::device_vector<Sphere<T> >& d_spheres,
    const Vector<3, T>& bot,
    const Vector<3, T>& top)
{
    thrust::device_vector<grace::uinteger64> d_keys(d_spheres.size());
    morton_keys_sph(d_spheres, bot, top, d_keys);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());
}

template <typename T>
GRACE_HOST void euclidean_deltas_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    thrust::device_vector<float>& d_deltas)
{
    compute_deltas(d_spheres, d_deltas,
                   DeltaEuclidean<const Sphere<T>*, CentroidSphere<T> >());
}

template <typename T>
GRACE_HOST void surface_area_deltas_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    thrust::device_vector<float>& d_deltas)
{
    compute_deltas(d_spheres, d_deltas,
                   DeltaSurfaceArea<const Sphere<T>*, AABBSphere>());
}

// KeyType should be grace::uinteger{32,64}.
// Note that the resulting deltas have the same type for this distance metric.
template <typename KeyType>
GRACE_HOST void XOR_deltas_sph(
    const thrust::device_vector<KeyType>& d_morton_keys,
    thrust::device_vector<KeyType>& d_deltas)
{
    compute_deltas(d_morton_keys, d_deltas, DeltaXOR());
}

template <typename T, typename DeltaType>
GRACE_HOST void ALBVH_sph(
    const thrust::device_vector<Sphere<T> >& d_spheres,
    const thrust::device_vector<DeltaType>& d_deltas,
    Tree& d_tree)
{
    build_ALBVH(d_tree, d_spheres, d_deltas, AABBSphere());
}

} // namespace grace
