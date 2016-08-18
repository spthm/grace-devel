// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/build_sph.cuh"
#include "grace/cuda/nodes.h"
#include "grace/cuda/trace_sph.cuh"
#include "grace/cuda/util/extrema.cuh"
#include "grace/ray.h"
#include "grace/sphere.h"
#include "helper/images.hpp"
#include "helper/tree.cuh"
#include "helper/rays.cuh"
#include "helper/read_gadget.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

typedef grace::Sphere<float> SphereType;

int main(int argc, char* argv[])
{
    size_t N_rays = 512 * 512;
    int max_per_leaf = 32;
    std::string fname = "../data/gadget/0128/Data_025";

    if (argc > 1)
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        max_per_leaf = (int)std::strtol(argv[2], NULL, 10);
    if (argc > 3) {
        fname = std::string(argv[3]);
    }

    size_t N_per_side = std::floor(std::pow(N_rays, 0.500001));
    // N_rays must be a multiple of 32.
    N_per_side = ((N_per_side + 32 - 1) / 32) * 32;
    N_rays = N_per_side * N_per_side;

    std::cout << "Gadget file:             " << fname << std::endl;
    // Vector is resized in read_gadget().
    thrust::device_vector<SphereType> d_spheres;
    read_gadget(fname, d_spheres);
    const size_t N = d_spheres.size();

    std::cout << "Number of particles:     " << N << std::endl
              << "Number of rays:          " << N_rays << std::endl
              << "Number of rays per side: " << N_per_side << std::endl
              << "Max particles per leaf:  " << max_per_leaf << std::endl
              << std::endl;


    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::Tree d_tree(N, max_per_leaf);

    // build_tree can compute the x/y/z limits for us, but we compute them
    // explicitly as we also need them for othogonal_rays_z.
    SphereType mins, maxs;
    grace::min_vec4(d_spheres, &mins);
    grace::max_vec4(d_spheres, &maxs);
    // orthogonal_rays_z() takes the maximum and minimum particle radii into
    // account - useful when integrating over the entire SPH field, but not
    // useful when creating images, as it leads to rays with zero integral-
    // values, and hence requires introduces a huge dynamic range.
    mins.r = maxs.r = 0;

    build_tree(d_spheres, mins, maxs, d_tree);
    orthogonal_rays_z(N_per_side, mins, maxs, d_rays);

    thrust::device_vector<float> d_integrals(N_rays);
    grace::trace_cumulative_sph(d_rays,
                                d_spheres,
                                d_tree,
                                d_integrals);

    float max_integral = thrust::reduce(d_integrals.begin(),
                                        d_integrals.end(),
                                        0.0f, thrust::maximum<float>());
    float min_integral = thrust::reduce(d_integrals.begin(),
                                        d_integrals.end(),
                                        1E20, thrust::minimum<float>());
    float mean_integral = thrust::reduce(d_integrals.begin(),
                                         d_integrals.end(),
                                         0.0f, thrust::plus<float>());
    mean_integral /= static_cast<float>(d_integrals.size());

    std::cout << "Mean output " << mean_integral << std::endl
              << "Max output: " << max_integral << std::endl
              << "Min output: " << min_integral << std::endl
              << std::endl;

    thrust::host_vector<float> h_integrals = d_integrals;
    // Increase the dynamic range.
    // Avoid zero so that max_integral - min_integral is a useful range
    // (required for the make_bitmap function).
    min_integral = std::max(1E-20f, min_integral);
    for (size_t i = 0; i < N_rays; ++i) {
        h_integrals[i] = std::log10(h_integrals[i]);
    }
    min_integral = std::log10(min_integral);
    max_integral = std::log10(max_integral);

    make_bitmap(thrust::raw_pointer_cast(h_integrals.data()),
                N_per_side, N_per_side,
                min_integral, max_integral,
                "density.bmp");

    return EXIT_SUCCESS;
}
