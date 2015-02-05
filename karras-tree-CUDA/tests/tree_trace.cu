// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/gen_rays.cuh"

int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    unsigned int N = 100000;
    unsigned int N_rays = 313*32; // = 10,016
    unsigned int max_per_leaf = 32;

    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (unsigned int) std::strtol(argv[2], NULL, 10);
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

    /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

    thrust::device_vector<float4> d_spheres_xyzr(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f) );


    /* Build the tree from the random data. */

    thrust::device_vector<unsigned int> d_keys(N);
    float3 top = make_float3(1.f, 1.f, 1.f);
    float3 bot = make_float3(0.f, 0.f, 0.f);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres_xyzr.begin());

    grace::Tree d_tree(N, max_per_leaf);
    thrust::device_vector<float> d_deltas(N+1);

    grace::compute_deltas(d_spheres_xyzr, d_deltas);
    grace::build_tree(d_tree, d_deltas, d_spheres_xyzr);
    grace::compact_tree(d_tree);

    // Keys no longer needed.
    d_keys.clear();
    d_keys.shrink_to_fit();


    /* Generate the rays (emitted from box centre and of length 2). */

    float ox, oy, oz, length;
    ox = oy = oz = 0.5f;
    length = 2;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);


    /* Trace for per-ray hit counts. */

    thrust::device_vector<unsigned int> d_hit_counts(N_rays);
    grace::trace_hitcounts(d_rays, d_hit_counts, d_tree,
                           d_spheres_xyzr);


    /* Loop through all rays and test for interestion with all particles
     * directly.
     */

    thrust::host_vector<float4> h_spheres_xyzr = d_spheres_xyzr;
    thrust::host_vector<grace::Ray> h_rays = d_rays;
    thrust::host_vector<unsigned int> h_hit_counts(N_rays);
    thrust::host_vector<unsigned int> h_d_hit_counts = d_hit_counts;

    for (unsigned int i=0; i<N_rays; i++)
    {
        grace::Ray ray = h_rays[i];
        // We use doubles to force the host to make double-precision floating-
        // point calculations in sphere_hit().
        //
        // nvcc automatically compiles to FMA where possible, which contains
        // only one round (compared to a+b -> round -> *c -> round, which has
        // two). In some edge cases, this results in different host and device
        // results for sphere_hit() if dummy1,2 are floats.
        // An alternative is to leave them as floats, but compile the code with
        // -fma=false (only available for CC >= 2.0); in this case host and
        // device agree, but both are inaccurate.
        double dummy1, dummy2;
        unsigned int hits = 0;

        for (unsigned int j=0; j<N; j++)
        {
            if (grace::sphere_hit(ray, h_spheres_xyzr[j], dummy1, dummy2))
                hits++;
        }

        h_hit_counts[i] = hits;
    }

    unsigned int failures = 0u;
    for (unsigned int i=0; i<N_rays; i++)
    {
        if (h_hit_counts[i] != h_d_hit_counts[i])
        {
            failures++;
            std::cout << "Trace failed for ray " << i << " (post-sort index):"
                      << std::endl;
            std::cout << "Direct test hits " << h_hit_counts[i] << "  |  "
                      << "Tree hits " << h_d_hit_counts[i] << std::endl;
            std::cout << std::endl;
        }
    }

    if (failures == 0)
    {
        std::cout << "All device tree-traced rays agree with direct interestion"
                  << " tests on host." << std::endl;
    }
    else
    {
        std::cout << std::endl;
        std::cout << failures << " intersections failed." << std::endl;
        std::cout << std::endl;
        std::cout << "Device code may compile to FMA instructions; try "
                  << "compiling this file" << std::endl;
        std::cout << "with nvcc's -fmad=false option."  << std::endl;
        std::cout << "Note that FMADs reduce rounding error, so should not in "
                  << "general be disabled." << std::endl;
    }

} // End device code.

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
