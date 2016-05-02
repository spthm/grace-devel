// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/ray.h"
#include "grace/cuda/util/extrema.cuh"
#include "grace/cuda/kernels/trace_sph.cuh"
#include "helper/rays.cuh"
#include "helper/read_gadget.cuh"
#include "helper/tree.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    size_t N_rays = 512 * 512;
    int max_per_leaf = 32;
    std::string fname = "../data/gadget/0128/Data_025";
    float tolerance = 5e-4;

    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        max_per_leaf = (int)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        fname = std::string(argv[3]);
    }
    if (argc > 4) {
        tolerance = (float)std::strtod(argv[4], NULL);
    }

    size_t N_per_side = std::floor(std::pow(N_rays, 0.500001));
    // N_rays must be a multiple of 32.
    N_per_side = ((N_per_side + 32 - 1) / 32) * 32;
    N_rays = N_per_side * N_per_side;


    std::cout << "Gadget file:               " << fname << std::endl;
    // Vector is resized in read_gadget().
    thrust::device_vector<float4> d_spheres;
    read_gadget(fname, d_spheres);
    size_t N = d_spheres.size();

    std::cout << "Number of particles:       " << N << std::endl
              << "Number of rays:            " << N_rays << std::endl
              << "Max particles per leaf:    " << max_per_leaf << std::endl
              << "Intergral error tolerance: " << tolerance << std::endl
              << std::endl;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::device_vector<float> d_integrals(N_rays);
    grace::Tree d_tree(N, max_per_leaf);

    // build_tree can compute the x/y/z limits for us, but we compute them
    // explicitly as we also need them for othogonal_rays_z.
    float4 mins, maxs;
    grace::min_vec4(d_spheres, &mins);
    grace::max_vec4(d_spheres, &maxs);


    float area_per_ray;

    build_tree(d_spheres, mins, maxs, d_tree);
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
