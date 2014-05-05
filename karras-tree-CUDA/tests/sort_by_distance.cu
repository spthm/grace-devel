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

    unsigned int N = 1000000;
    // Few because the random spheres result in many hits per ray.
    unsigned int N_rays = 25000;

    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = (unsigned int) std::strtol(argv[2], NULL, 10);
    }

    std::cout << "Generating " << N << " random points and " << N_rays
              << " random rays." << std::endl;
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

    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);

    // Working arrays no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();


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
    thrust::device_vector<float> d_trace_dists;

    grace::trace<float>(d_rays, d_traced_rho, d_trace_dists,
                        d_nodes, d_spheres_xyzr, d_rho);

    unsigned int total_hits = d_traced_rho.size();
    std::cout << "Total hits:   " << total_hits << std::endl;
    std::cout << "Mean per ray: " << ((float)total_hits) / N_rays << std::endl;
    std::cout << std::endl;


    /* Verify the intersection data has been correctly sorted by intersection
     * distance.
     */

    thrust::host_vector<float> h_trace_dists = d_trace_dists;
    // We require the per-ray offsets into d_trace_dists/rho that grace::trace
    // computes internally. Simply (wastefully) compute them again here.
    thrust::device_vector<unsigned int> d_hit_offsets(N_rays);
    grace::trace_hitcounts(d_rays, d_hit_offsets, d_nodes, d_spheres_xyzr);
    thrust::exclusive_scan(d_hit_offsets.begin(), d_hit_offsets.end(),
                           d_hit_offsets.begin());
    thrust::host_vector<unsigned int> h_hit_offsets = d_hit_offsets;

    bool success = true;
    for (int ray_i=0; ray_i<N_rays; ray_i++) {
        int start = h_hit_offsets[ray_i];
        int end = (ray_i < N_rays-1 ? h_hit_offsets[ray_i+1] : total_hits);

        float dist = h_trace_dists[start];

        for (int hit_i=start+1; hit_i<end; hit_i++) {
            float next = h_trace_dists[hit_i];
            if (next < dist){
                std::cout << "Error for ray " << ray_i << "!  distance["
                          << hit_i << "] = " << std::setw(8) << next
                          << " < distance[" << hit_i - 1 << "] = "
                          << std::setw(8) << dist << std::endl;
                std::cout << std::endl;
                success = false;
            }
            dist = next;
        }

    }

    if (success) {
        std::cout << "All " << N_rays << " rays sorted correctly."
                  << std::endl;
    }

} // End device code.

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
