// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include <cstdlib>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "../../nodes.h"
#include "../../ray.h"
#include "../../utils.cuh"
#include "../../kernels/morton.cuh"
#include "../../kernels/bintree_build.cuh"
#include "../../kernels/bintree_trace.cuh"
#include "../../kernels/gen_rays.cuh"

int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    size_t N = 2000000;
    size_t N_rays = 32 * 1000; // = 32,000
    unsigned int max_per_leaf = 32;

    if (argc > 1) {
        N = (size_t) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (size_t) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        max_per_leaf = (unsigned int) std::strtol(argv[3], NULL, 10);
    }

    std::cout << "Testing " << N << " random points and " << N_rays
              << " random rays, with up to " << max_per_leaf << " point(s) per"
              << std::endl
              << "leaf." << std::endl;
    std::cout << std::endl;

{ // Device code.

    /* Generate random points. */

    float min_radius = 80.f;
    float max_radius = 400.f;
    thrust::device_vector<float4> d_spheres(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres.begin(),
                      grace::random_float4_functor(-1E4f, 1E4f,
                                                   min_radius, max_radius));


    /* Build the tree from the random data. */

    thrust::device_vector<unsigned int> d_keys(N);
    float3 top = make_float3(1.2E4f, 1.2E4f, 1.2E4f);
    float3 bot = make_float3(-1.2E4f, -1.2E4f, -1.2E4f);

    grace::morton_keys(d_keys, d_spheres, top, bot);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres.begin());

    grace::Tree d_tree(N, max_per_leaf);
    thrust::device_vector<float> d_deltas(N+1);

    grace::compute_deltas(d_spheres, d_deltas);
    grace::build_tree(d_tree, d_spheres, d_deltas, d_spheres);

    // Keys no longer needed.
    d_keys.clear();
    d_keys.shrink_to_fit();


    /* Generate the rays (emitted from box centre and of length 2E4). */

    float ox, oy, oz, length;
    ox = oy = oz = 0.0f;
    length = 2E4f;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);


    /* Trace for per-ray hit counts. */

    thrust::device_vector<int> d_hit_counts(N_rays);
    grace::trace_hitcounts(d_rays, d_hit_counts, d_tree, d_spheres);

    double total = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end());
    double mean_hits = static_cast<double>(total) / N_rays;



    /* Loop through all rays and test for interestion with all particles
     * directly.
     */

    thrust::host_vector<float4> h_spheres = d_spheres;
    thrust::host_vector<grace::Ray> h_rays = d_rays;
    thrust::host_vector<int> h_hit_counts(N_rays);

    #pragma omp parallel for
    for (size_t ri = 0; ri < N_rays; ++ri)
    {
        grace::Ray ray = h_rays[ri];
        int hits = 0;
        float b2, d;

        for (size_t si = 0; si < N; ++si) {
            if (grace::sphere_hit(ray, h_spheres[si], b2, d)) {
                hits++;
            }
        }

        h_hit_counts[ri] = hits;
    }


    /* Compare all CPU-computed per-ray hit counts to those computed using the
     * tree on the GPU.
     */

    thrust::host_vector<int> h_d_hit_counts = d_hit_counts;
    size_t failed_rays = 0;
    size_t failed_intersections = 0;
    for (size_t ri = 0; ri < N_rays; ++ri)
    {
        if (h_hit_counts[ri] != h_d_hit_counts[ri])
        {
            failed_rays += 1;
            failed_intersections = abs(h_hit_counts[ri] - h_d_hit_counts[ri]);

            std::cout << "FAILED trace for ray " << ri << ":"
                      << std::endl;
            std::cout << "  CPU hits: " << h_hit_counts[ri] << std::endl;
            std::cout << "  GPU hits: " << h_d_hit_counts[ri] << std::endl;
            std::cout << std::endl;
        }
    }

    if (failed_rays == 0)
    {
        std::cout << "PASSED" << std::endl;
        std::cout << "All device tree-traced rays agree with direct ray-sphere "
                  << "intersection tests on host." << std::endl;
    }
    else
    {
        std::cout << "FAILED " << failed_intersections << " intersection tests "
                  << "in " << failed_rays << " rays."
                  << std::endl;
    }
    std::cout << "Mean of " << mean_hits << " hits per ray (device)."
              << std::endl;


} // End device code.

    // Exit cleanly.
    cudaDeviceReset();
    return 0;
}
