#include "ray.cuh"
#include "grace/cuda/generate_rays.cuh"
#include "grace/cuda/prngstates.cuh"

#include <thrust/device_vector.h>

void isotropic_rays(
    thrust::host_vector<Ray>& h_rays,
    grace::Vector<3, float> origin,
    float length)
{
    const size_t N_rays = h_rays.size();

    grace::PrngStates rng_states;
    thrust::device_vector<Ray> d_rays(N_rays);
    // Ray is a typedef for grace::Ray.
    grace::uniform_random_rays(origin, length, rng_states, d_rays);

    h_rays = d_rays;
}
