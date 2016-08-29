// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/trace_sph.cuh"
#include "grace/ray.h"
#include "grace/sphere.h"
#include "grace/vector.h"
#include "helper/tree.cuh"
#include "helper/rays.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

typedef grace::Sphere<float> SphereType;

void two_spheres(const SphereType mins, const SphereType maxs,
                 thrust::device_vector<SphereType>& d_spheres)
{
    thrust::host_vector<SphereType> h_spheres(2);

    float radius = (mins.r + maxs.r) / 2.0;
    grace::Vector<3, float> mid;
    mid.x = (mins.x + maxs.x) / 2.0;
    mid.y = (mins.y + maxs.y) / 2.0;
    mid.z = (mins.z + maxs.z) / 2.0;

    h_spheres[0].x = (mins.x + mid.x) / 2.0;
    h_spheres[0].y = (mins.y + mid.y) / 2.0;
    h_spheres[0].z = (mins.z + mid.z) / 2.0;
    h_spheres[0].r = radius;

    h_spheres[1].x = (maxs.x + mid.x) / 2.0;
    h_spheres[1].y = (maxs.y + mid.y) / 2.0;
    h_spheres[1].z = (maxs.z + mid.z) / 2.0;
    h_spheres[1].r = radius;

    d_spheres = h_spheres;
}

int main(int argc, char* argv[])
{
    // Tree does not work for N < 2 objects.
    const size_t N = 2;
    // DO NOT CHANGE. There are only two spheres.
    const int maxs_per_leaf = 1;

    size_t N_rays = 512 * 512;
    float tolerance = 5e-4;

    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        tolerance = (float)std::strtod(argv[2], NULL);
    }

    size_t N_per_side = floor(pow(N_rays, 0.500001));
    // N_rays must be a multiple of 32.
    N_per_side = ((N_per_side + 32 - 1) / 32) * 32;
    N_rays = N_per_side * N_per_side;

    std::cout << "Number of particles:       " << N << std::endl
              << "Number of rays:            " << N_rays << std::endl
              << "Intergral error tolerance: " << tolerance << std::endl
              << std::endl;

    // Allocate permanent vectors before temporaries.
    thrust::device_vector<SphereType> d_spheres(N);
    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::device_vector<float> d_integrals(N_rays);
    grace::Tree d_tree(N, maxs_per_leaf);

    float area_per_ray;
    SphereType mins, maxs;
    maxs.x = maxs.y = maxs.z = 1.f;
    mins.x = mins.y = mins.z = -1.f;
    maxs.r = mins.r = 0.2f;

    two_spheres(mins, maxs, d_spheres);
    build_tree(d_spheres,
               grace::Vector<3, float>(mins), grace::Vector<3, float>(maxs),
               d_tree);
    plane_parallel_rays_z(N_per_side, mins, maxs, d_rays, &area_per_ray);
    grace::trace_cumulative_sph(d_rays, d_spheres, d_tree, d_integrals);

    // ~ Integrate over x and y.
    float integrated_sum = thrust::reduce(d_integrals.begin(),
                                          d_integrals.end(),
                                          0.0f, thrust::plus<float>());
    // Multiply by the pixel area to complete the x-y integration.
    integrated_sum *= area_per_ray;
    // Correct integration implies integrated_sum == N_particles.
    integrated_sum /= static_cast<float>(N);

    std::cout << "Normalized volume integral: " << integrated_sum
              << std::endl;

    return abs(1.0 - integrated_sum) < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
