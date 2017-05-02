// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/generate_rays.cuh"
#include "grace/cuda/prngstates.cuh"
#include "grace/cuda/trace_sph.cuh"
#include "grace/generic/intersect.h"
#include "grace/ray.h"
#include "grace/sphere.h"
#include "grace/vector.h"
#include "helper/tree.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <cstdlib>
#include <iostream>

typedef grace::Sphere<float> SphereType;

int main(int argc, char* argv[])
{
    size_t N = 2000000;
    size_t N_rays = 32 * 1000; // = 32,000
    int max_per_leaf = 32;
    bool verbose = true;

    if (argc > 1) {
        N = (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (size_t)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        max_per_leaf = (int)std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        verbose = (std::string(argv[4]) == "true") ? true : false;
    }

    std::cout << "Number of particles:     " << N << std::endl
              << "Number of rays:          " << N_rays << std::endl
              << "Max particles per leaf:  " << max_per_leaf << std::endl
              << std::endl;

    thrust::device_vector<SphereType> d_spheres(N);
    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::device_vector<int> d_hit_counts(N_rays);
    grace::Tree d_tree(N, max_per_leaf);
    grace::PrngStates rng_states;

    SphereType low = SphereType(-1E4f, -1E4f, -1E4f, 80.f);
    SphereType high = SphereType(1E4f, 1E4f, 1E4f, 400.f);
    random_spheres_tree(low, high, N, d_spheres, d_tree);
    grace::uniform_random_rays(grace::Vector<3, float>(), 2E4f, rng_states,
                               d_rays);

    grace::trace_hitcounts_sph(d_rays, d_spheres, d_tree, d_hit_counts);
    double total = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end());
    double mean_hits = total / N_rays;
    std::cout << "Mean of " << mean_hits << " hits per ray (device)."
              << std::endl << std::endl;

    thrust::host_vector<SphereType> h_spheres = d_spheres;
    thrust::host_vector<grace::Ray> h_rays = d_rays;
    thrust::host_vector<int> ref_hit_counts(N_rays);
    #pragma omp parallel for
    for (size_t ri = 0; ri < N_rays; ++ri)
    {
        grace::Ray ray = h_rays[ri];
        int hits = 0;
        float b2, d;

        for (size_t si = 0; si < N; ++si) {
            if (grace::sphere_hit(ray, h_spheres[si], b2, d)) {
                ++hits;
            }
        }

        ref_hit_counts[ri] = hits;
    }

    thrust::host_vector<int> h_hit_counts = d_hit_counts;
    size_t failed_rays = 0;
    size_t failed_intersections = 0;
    for (size_t ri = 0; ri < N_rays; ++ri)
    {
        if (ref_hit_counts[ri] != h_hit_counts[ri])
        {
            ++failed_rays;
            failed_intersections += abs(ref_hit_counts[ri] - h_hit_counts[ri]);

            if (!verbose) {
                continue;
            }

            std::cout << "FAILED trace for ray " << ri << ":"
                      << std::endl;
            std::cout << "  CPU hits: " << ref_hit_counts[ri] << std::endl;
            std::cout << "  GPU hits: " << h_hit_counts[ri] << std::endl;
            std::cout << std::endl;
        }
    }

    if (verbose) {
        std::cout << std::endl;
    }

    if (failed_rays == 0)
    {
        std::cout << "PASSED" << std::endl;
    }
    else
    {
        std::cout << "FAILED" << std::endl
                  << failed_intersections << " intersection test"
                  << (failed_intersections > 1 ? "s " : " ")
                  << "failed over " << failed_rays << " ray"
                  << (failed_rays > 1 ? "s." : ".")
                  << std::endl;
    }

    return failed_rays == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
