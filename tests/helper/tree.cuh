#include "utils.cuh"
#include "kernels/build_sph.cuh"
#include "util/meta.h"

#include "helper/random.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

// Always uses 30-bit keys.
template <typename Real4>
void random_spheres_tree(const Real4 high, const Real4 low, const size_t N,
                         thrust::device_vector<Real4>& spheres,
                         grace::Tree& tree)
{
    typedef typename grace::Real4ToRealMapper<Real4>::type Real;

    spheres.resize(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      spheres.begin(),
                      random_real4_functor<Real4>(low, high) );

    thrust::device_vector<Real> deltas(N + 1);

    const float3 bottom = make_float3(low.x, low.y, low.z);
    const float3 top = make_float3(high.x, high.y, high.z);

    grace::morton_keys30_sort_sph(spheres, top, bottom);
    grace::euclidean_deltas_sph(spheres, deltas);
    grace::ALBVH_sph(spheres, deltas, tree);
}
