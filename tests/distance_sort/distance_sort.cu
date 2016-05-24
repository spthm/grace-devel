// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/gen_rays.cuh"
#include "grace/ray.h"
#include "helper/tree.cuh"
#include "helper/trace.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int verify_intersection_order(const thrust::host_vector<int>& offsets,
                              const thrust::host_vector<float>& distances,
                              const bool verbose = true)
{
    const size_t total_hits = distances.size();
    const size_t N_rays = offsets.size();

    int failures = 0;
    for (size_t i = 0; i < N_rays; ++i)
    {
        int start = offsets[i];
        int end = (i < N_rays - 1 ? offsets[i + 1] : total_hits);

        if (start == end) {
            // Ray hit nothing.
            continue;
        }

        // Check first hit is not < 0 along ray.
        float dist = distances[start];
        if (dist < 0 && verbose) {
            ++failures;
            std::cout << "Error @ ray " << i << "!" << std::endl
                      << "  First particle hit distance = " << std::setw(8)
                      << dist << std::endl
                      << std::endl;
        }

        // First hit along ray == 0 is usually a sign of a bug, and if all
        // distances are zero the next loop will not catch it.
        if (dist == 0 && verbose) {
            std::cout << "Warning @ ray " << i << std::endl
                      << "  First particle hit distance = 0; check particle "
                      << "hit distances are not all zero." << std::endl
                      << std::endl;
        }

        for (int j = start + 1; j < end; ++j)
        {
            float next_dist = distances[j];

            if (next_dist < dist && verbose) {
                ++failures;
                std::cout << "Error @ ray " << i << "!" << std::endl
                          << "  distance[" << j << "] = " << std::setw(8)
                          << next_dist
                          << " < distance[" << j - 1 << "] = " << std::setw(8)
                          << dist
                          << std::endl
                          << std::endl;
            }

            dist = next_dist;
        }
    }

    return failures;
}

int main(int argc, char* argv[]) {

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    /* Initialize run parameters. */

    size_t N = 100000;
    // Relatively few because the random spheres result in many hits per ray.
    size_t N_rays = 2500 * 32; // = 80,000.
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
        verbose = (std::string(argv[3]) == "true") ? true : false;
    }

    std::cout << "Number of particles:    " << N << std::endl
              << "Number of rays:         " << N_rays << std::endl
              << "Max particles per leaf: " << max_per_leaf << std::endl
              << std::endl;

    // Allocate permanent vectors before temporaries.
    thrust::device_vector<float4> d_spheres(N);
    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::host_vector<int> h_ray_offsets(N_rays);
    grace::Tree d_tree(N, max_per_leaf);
    thrust::host_vector<float> h_distances; // Will be resized.

    // Random spheres in [0, 1) are generated, with radii in [0, 0.1).
    float4 high = make_float4(1.f, 1.f, 1.f, 0.1f);
    float4 low = make_float4(0.f, 0.f, 0.f, 0.f);
    // Rays emitted from box centre and of sufficient length to exit the box.
    float4 O = make_float4(.5f, .5f, .5f, 2.f);

    random_spheres_tree(low, high, N, d_spheres, d_tree);
    grace::uniform_random_rays(d_rays, O.x, O.y, O.z, O.w);
    trace_distances(d_rays, d_spheres, d_tree, h_ray_offsets, h_distances);

    int failures = verify_intersection_order(h_ray_offsets, h_distances,
                                             verbose);

    if (failures == 0) {
        std::cout << "All " << N_rays << " rays sorted correctly."
                  << std::endl;
    }
    else {
        std::cout << failures << " intersections sorted incorrectly."
                  << std::endl;
    }

    size_t total_hits = h_distances.size();
    std::cout << std::endl
              << "Total hits:   " << total_hits << std::endl
              << "Mean per ray: " << static_cast<float>(total_hits) / N_rays
              << std::endl
              << std::endl;

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
