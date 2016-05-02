#pragma once

#include "grace/cuda/nodes.h"
#include "grace/cuda/kernels/build_sph.cuh"
#include "grace/generic/util/meta.h"

#include "helper/random.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

// Always uses 30-bit keys.
template <typename Real4>
void build_tree(thrust::device_vector<Real4>& spheres,
                grace::Tree& tree)
{
    typedef typename grace::Real4ToRealMapper<Real4>::type Real;

    thrust::device_vector<Real> deltas(spheres.size() + 1);

    grace::morton_keys30_sort_sph(spheres);
    grace::euclidean_deltas_sph(spheres, deltas);
    grace::ALBVH_sph(spheres, deltas, tree);
}

// Always uses 30-bit keys.
// low and high can be any type with .x/.y/.z components.
template <typename Real3, typename Real4>
void build_tree(thrust::device_vector<Real4>& spheres,
                const Real3 low, const Real3 high,
                grace::Tree& tree)
{
    typedef typename grace::Real4ToRealMapper<Real4>::type Real;

    const float3 bottom = make_float3(low.x, low.y, low.z);
    const float3 top = make_float3(high.x, high.y, high.z);
    thrust::device_vector<Real> deltas(spheres.size() + 1);

    grace::morton_keys30_sort_sph(spheres, top, bottom);
    grace::euclidean_deltas_sph(spheres, deltas);
    grace::ALBVH_sph(spheres, deltas, tree);
}

// Always uses 30-bit keys.
template <typename Real4>
void random_spheres_tree(const Real4 low, const Real4 high, const size_t N,
                         thrust::device_vector<Real4>& spheres,
                         grace::Tree& tree)
{
    spheres.resize(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      spheres.begin(),
                      random_real4_functor<Real4>(low, high) );

    build_tree(spheres, low, high, tree);
}
