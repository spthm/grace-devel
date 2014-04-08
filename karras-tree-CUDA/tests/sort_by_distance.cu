#include <cmath>
#include <sstream>
#include <iomanip>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "utils.cuh"
#include "../kernel_config.h"
#include "../nodes.h"
#include "../ray.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"

int main(int argc, char* argv[]) {

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    /* Initialize run parameters. */

    unsigned int N = 1000000;
    unsigned int N_rays = 10000;

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

    float4 bot = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 top = make_float4(1.f, 1.f, 1.f, 0.f);

    // One set of keys for sorting spheres' x, y, z and radii, another for
    // sorting their densities.
    thrust::device_vector<grace::uinteger32> d_keys(N);
    thrust::device_vector<grace::uinteger32> d_keys_2(N);

    grace::morton_keys(d_spheres_xyzr, d_keys, bot, top);
    thrust::copy(d_keys.begin(), d_keys.end(), d_keys_2.begin());

    thrust::sort_by_key(d_keys_2.begin(), d_keys_2.end(),
                        d_spheres_xyzr.begin());
    d_keys_2.clear(); d_keys_2.shrink_to_fit();

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_rho.begin());

    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);

    // Working arrays no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();


    /* Generate the rays, emitted emitted from box centre (.5, .5, .5) and of
     * sufficient length to be terminated outside the box.
     */

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<unsigned int> h_ray_keys(N_rays);

    thrust::host_vector<float> h_dxs(N_rays);
    thrust::host_vector<float> h_dys(N_rays);
    thrust::host_vector<float> h_dzs(N_rays);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N_rays),
                      h_dxs.begin(),
                      grace::random_float_functor(0u, -1.0f, 1.0f) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N_rays),
                      h_dys.begin(),
                      grace::random_float_functor(1u, -1.0f, 1.0f) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N_rays),
                      h_dzs.begin(),
                      grace::random_float_functor(2u, -1.0f, 1.0f) );

    float length = sqrt(3);

    for (int i=0; i<N_rays; i++) {
        float N_dir = sqrt(h_dxs[i]*h_dxs[i] +
                           h_dys[i]*h_dys[i] +
                           h_dzs[i]*h_dzs[i]);

        h_rays[i].dx = h_dxs[i] / N_dir;
        h_rays[i].dy = h_dys[i] / N_dir;
        h_rays[i].dz = h_dzs[i] / N_dir;

        h_rays[i].ox = h_rays[i].oy = h_rays[i].oz = 0.5f;

        h_rays[i].length = length;

        h_rays[i].dclass = 0;
        if (h_dxs[i] >= 0)
            h_rays[i].dclass += 1;
        if (h_dys[i] >= 0)
            h_rays[i].dclass += 2;
        if (h_dzs[i] >= 0)
            h_rays[i].dclass += 4;

        // Floats must be in (0, 1) for morton_key().
        h_ray_keys[i] = grace::morton_key((h_rays[i].dx+1)/2.f,
                                          (h_rays[i].dy+1)/2.f,
                                          (h_rays[i].dz+1)/2.f);
    }
    h_dxs.clear();
    h_dxs.shrink_to_fit();
    h_dys.clear();
    h_dys.shrink_to_fit();
    h_dzs.clear();
    h_dxs.shrink_to_fit();


    /* Perform a full trace and verify the intersection data has been correctly
     * sorted by intersection distance.
     */

    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<unsigned int> d_ray_keys = h_ray_keys;

    thrust::sort_by_key(d_ray_keys.begin(), d_ray_keys.end(),
                        d_rays.begin());

    grace::KernelIntegrals<float> lookup;
    thrust::device_vector<float> d_b_integrals(&lookup.table[0],
                                               &lookup.table[50]);

    thrust::device_vector<unsigned int> d_hit_offsets(N_rays);

    grace::gpu::trace_hitcounts_kernel<<<28, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        d_rays.size(),
        thrust::raw_pointer_cast(d_hit_offsets.data()),
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        d_nodes.hierarchy.size(),
        thrust::raw_pointer_cast(d_spheres_xyzr.data()));

    // Allocate output array based on per-ray hit counts, and calculate
    // individual ray offsets into this array.
    unsigned int last_ray_hits = d_hit_offsets[N_rays-1];
    thrust::exclusive_scan(d_hit_offsets.begin(), d_hit_offsets.end(),
                           d_hit_offsets.begin());
    unsigned int total_hits = d_hit_offsets[N_rays-1] + last_ray_hits;

    std::cout << "Total hits:   " << total_hits << std::endl;
    std::cout << "Mean per ray: " << ((float)total_hits) / N_rays << std::endl;

    thrust::device_vector<float> d_traced_rho(total_hits);
    thrust::device_vector<float> d_trace_dists(total_hits);

    grace::gpu::trace_kernel<<<28, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        d_rays.size(),
        thrust::raw_pointer_cast(d_traced_rho.data()),
        thrust::raw_pointer_cast(d_trace_dists.data()),
        thrust::raw_pointer_cast(d_hit_offsets.data()),
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        d_nodes.hierarchy.size(),
        thrust::raw_pointer_cast(d_spheres_xyzr.data()),
        thrust::raw_pointer_cast(d_rho.data()),
        thrust::raw_pointer_cast(d_b_integrals.data()));


    /* Sort and verify. */


    thrust::device_vector<unsigned int> d_ray_segments(d_trace_dists.size());
    thrust::constant_iterator<unsigned int> first(1);
    thrust::constant_iterator<unsigned int> last = first + d_hit_offsets.size();

    thrust::scatter(first, last,
                    d_hit_offsets.begin(),
                    d_ray_segments.begin());

    thrust::inclusive_scan(d_ray_segments.begin(), d_ray_segments.end(),
                           d_ray_segments.begin());

    thrust::sort_by_key(d_trace_dists.begin(), d_trace_dists.end(),
                        thrust::make_zip_iterator(
                            thrust::make_tuple(d_traced_rho.begin(),
                                               d_ray_segments.begin())
                        )
    );
    thrust::stable_sort_by_key(d_ray_segments.begin(), d_ray_segments.end(),
                               thrust::make_zip_iterator(
                                   thrust::make_tuple(d_trace_dists.begin(),
                                                      d_traced_rho.begin())
                               )
    );

    thrust::host_vector<unsigned int> h_hit_offsets = d_hit_offsets;
    thrust::host_vector<float> h_trace_dists = d_trace_dists;

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
