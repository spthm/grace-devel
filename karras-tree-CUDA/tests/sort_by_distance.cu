// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include <cmath>
#include <sstream>
#include <iomanip>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/gen_rays.cuh"
#include "../kernels/sort.cuh"

int main(int argc, char* argv[]) {

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    /* Initialize run parameters. */

    unsigned int N = 100000;
    // Relatively few because the random spheres result in many hits per ray.
    unsigned int N_rays = 2500*32; // = 80,000.
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

    std::cout << "Generating " << N << " random points and " << N_rays
              << " random rays, with up to " << max_per_leaf << " point(s) per"
              << std::endl
              << "leaf." << std::endl;
    std::cout << std::endl;


{ // Device code.

    /* Generate N random points as floats in [0,1), and radii and densities
     * in [0,0.1).
     */

    thrust::device_vector<float4> d_spheres_xyzr(N);
    thrust::device_vector<float> d_rho(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_rho.begin(),
                      grace::random_float_functor(0.1f));


    /* Build the tree. */

    float3 top = make_float3(1.f, 1.f, 1.f);
    float3 bot = make_float3(0.f, 0.f, 0.f);

    thrust::device_vector<grace::uinteger32> d_keys(N);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
    grace::sort_by_key(d_keys, d_spheres_xyzr, d_rho);

    grace::Tree d_tree(N, max_per_leaf);
    thrust::device_vector<float> d_deltas(N+1);

    grace::compute_deltas(d_spheres_xyzr, d_deltas);
    grace::build_tree(d_tree, d_spheres_xyzr, d_deltas, d_spheres_xyzr);

    // Working arrays no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();
    d_deltas.clear(); d_deltas.shrink_to_fit();


    /* Generate the rays, emitted emitted from box centre (.5, .5, .5) and of
     * sufficient length to be terminated outside the box.
     */

    float ox, oy, oz, length;
    ox = oy = oz = 0.5f;
    length = sqrt(3);

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);


    /* Perform a full trace. */

    thrust::device_vector<float> d_traced_rho;
    thrust::device_vector<unsigned int> d_ray_offsets(N_rays);
    thrust::device_vector<unsigned int> d_hit_indices;
    thrust::device_vector<float> d_hit_distances;

    grace::trace<float>(d_rays,
                        d_traced_rho,
                        d_ray_offsets,
                        d_hit_indices,
                        d_hit_distances,
                        d_tree,
                        d_spheres_xyzr,
                        d_rho);

    thrust::device_vector<unsigned int> d_ray_segments(d_hit_indices.size());
    grace::offsets_to_segments(d_ray_offsets, d_ray_segments);
    grace::sort_by_distance(d_hit_distances,
                            d_ray_segments,
                            d_hit_indices,
                            d_traced_rho);

    unsigned int total_hits = d_traced_rho.size();
    std::cout << "Total hits:   " << total_hits << std::endl;
    std::cout << "Mean per ray: " << ((float)total_hits) / N_rays << std::endl;
    std::cout << std::endl;


    /* Verify the intersection data has been correctly sorted by intersection
     * distance.
     */

    thrust::host_vector<unsigned int> h_ray_offsets = d_ray_offsets;
    thrust::host_vector<float> h_hit_distances = d_hit_distances;

    unsigned int failures = 0u;
    for (int ray_i=0; ray_i<N_rays; ray_i++) {
        int start = h_ray_offsets[ray_i];
        int end = (ray_i < N_rays-1 ? h_ray_offsets[ray_i+1] : total_hits);

        if (start == end) {
            // Ray hit nothing. Move to next ray.
            continue;
        }

        // Check first hit is not < 0 along ray.
        float dist = h_hit_distances[start];
        if (dist < 0) {
            std::cout << "Error @ ray " << ray_i << "!" << std::endl;
            std::cout << "First particle hit distance = " << std::setw(8)
                      << dist << std::endl;
            std::cout << std::endl;
            failures++;
        }

        // Check first hit is not = 0 along ray; this is usually a sign of
        // a bug, and if all distances are zero, the next loop will not catch
        // it.
        if (dist == 0) {
            std::cout << "Warning @ ray " << ray_i << std::endl;
            std::cout << "First particle hit distance = 0; check all particle "
                      << "hit distances are not zero." << std::endl;
            std::cout << std::endl;
        }

        // Check all following hits are >= along the ray than those preceding.
        for (int i=start+1; i<end; i++) {
            float next_dist = h_hit_distances[i];

            if (next_dist < dist) {
                std::cout << "Error @ ray " << ray_i << "!" << std::endl;
                std::cout << "  distance[" << i << "] = " << std::setw(8)
                          << next_dist
                          << " < distance[" << i-1 << "] = " << std::setw(8)
                          << dist
                          << std::endl;
                std::cout << std::endl;
                failures++;
            }
            dist = next_dist;
        }

    }

    if (failures == 0u) {
        std::cout << "All " << N_rays << " rays sorted correctly."
                  << std::endl;
    }
    else {
        std::cout << failures << " intersections sorted incorrectly."
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Device code may compile to FMA instructions; if the "
                  << "above errors are within" << std::endl;
        std::cout << "floating point error, try compiling this file with "
                  << "nvcc's -fmad=false option."  << std::endl;
        std::cout << "Note that FMADs reduce rounding error, so should not in "
                  << "general be disabled." << std::endl;
    }

} // End device code.

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
