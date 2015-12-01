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

#include "nodes.h"
#include "ray.h"
#include "utils.cuh"
#include "kernels/build_sph.cuh"
#include "kernels/gen_rays.cuh"
#include "kernels/sort.cuh"
#include "kernels/trace_sph.cuh"

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

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f) );


    /* Build the tree. */

    float3 top = make_float3(1.f, 1.f, 1.f);
    float3 bot = make_float3(0.f, 0.f, 0.f);

    // Allocate permanent vectors before temporaries.
    grace::Tree d_tree(N, max_per_leaf);
    thrust::device_vector<float> d_deltas(N + 1);

    grace::morton_keys30_sort_sph(d_spheres_xyzr, top, bot);
    grace::euclidean_deltas_sph(d_spheres_xyzr, d_deltas);
    grace::ALBVH_sph(d_spheres_xyzr, d_deltas, d_tree);

    // Deltas no longer needed.
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

    thrust::device_vector<float> d_traced_integrals;
    thrust::device_vector<int> d_ray_offsets(N_rays);
    thrust::device_vector<int> d_hit_indices;
    thrust::device_vector<float> d_hit_distances;

    grace::trace_sph(d_rays,
                     d_spheres_xyzr,
                     d_tree,
                     d_ray_offsets,
                     d_hit_indices,
                     d_traced_integrals,
                     d_hit_distances);

    grace::sort_by_distance(d_hit_distances,
                            d_ray_offsets,
                            d_hit_indices,
                            d_traced_integrals);

    unsigned int total_hits = d_traced_integrals.size();
    std::cout << "Total hits:   " << total_hits << std::endl;
    std::cout << "Mean per ray: " << static_cast<float>(total_hits) / N_rays
              << std::endl;
    std::cout << std::endl;


    /* Verify the intersection data has been correctly sorted by intersection
     * distance.
     */

    thrust::host_vector<int> h_ray_offsets = d_ray_offsets;
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
            std::cout << "First particle hit distance = 0; check particle hit "
                      << "distances are not all zero." << std::endl;
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