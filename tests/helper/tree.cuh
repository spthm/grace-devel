#pragma once

#include "grace/cuda/build_sph.cuh"
#include "grace/cuda/nodes.h"
#include "grace/generic/meta.h"
#include "grace/sphere.h"

#include "helper/random.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

// Always uses 30-bit keys.
template <typename T>
void build_tree(thrust::device_vector<grace::Sphere<T> >& spheres,
                grace::Tree& tree)
{
    thrust::device_vector<T> deltas(spheres.size() + 1);

    grace::morton_keys30_sort_sph(spheres);
    grace::euclidean_deltas_sph(spheres, deltas);
    grace::ALBVH_sph(spheres, deltas, tree);
}

// Always uses 30-bit keys.
// low and high can be any type with .x/.y/.z components.
template <typename Real3, typename T>
void build_tree(thrust::device_vector<grace::Sphere<T> >& spheres,
                const Real3 low, const Real3 high,
                grace::Tree& tree)
{
    const float3 bottom = make_float3(low.x, low.y, low.z);
    const float3 top = make_float3(high.x, high.y, high.z);
    thrust::device_vector<T> deltas(spheres.size() + 1);

    grace::morton_keys30_sort_sph(spheres, bottom, top);
    grace::euclidean_deltas_sph(spheres, deltas);
    grace::ALBVH_sph(spheres, deltas, tree);
}

// Always uses 30-bit keys.
template <typename T>
void random_spheres_tree(const grace::Sphere<T> low,
                         const grace::Sphere<T> high,
                         const size_t N,
                         thrust::device_vector<grace::Sphere<T> >& spheres,
                         grace::Tree& tree)
{
    spheres.resize(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      spheres.begin(),
                      random_sphere_functor<grace::Sphere<T> >(low, high) );

    build_tree(spheres, low, high, tree);
}
