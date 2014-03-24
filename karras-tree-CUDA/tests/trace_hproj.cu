#include <cmath>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "utils.cuh"
#include "../types.h"
#include "../nodes.h"
#include "../ray.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"

int main(int argc, char* argv[])
{

    unsigned int N_rays = 250000;
    if (argc > 1) {
        N_rays = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    float N_rays_side = floor(pow(N_rays, 0.500001));

    // Generate two spheres (tree does not work for N < 2 objects).
    thrust::host_vector<float4> h_spheres_xyzr(2);
    size_t N = h_spheres_xyzr.size();
    h_spheres_xyzr[0].x = -0.5f;
    h_spheres_xyzr[0].y = 0.0f;
    h_spheres_xyzr[0].z = 0.0f;
    h_spheres_xyzr[0].w = 0.2f;
    h_spheres_xyzr[1].x = +0.5f;
    h_spheres_xyzr[1].y = 0.0f;
    h_spheres_xyzr[1].z = 0.0f;
    h_spheres_xyzr[1].w = 0.2f;

    // Set all masses equal to one - a pseudomass.
    thrust::host_vector<float> h_pmasses(N);
    for (int i=0; i<N; i++) {
        h_pmasses[i] = 1.0f;
    }

    std::cout << "Will trace " << N_rays << " rays through " << N
              << " particles..." << std::endl;
    std::cout << std::endl;

// Device code.
{
    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
    thrust::device_vector<float> d_pmasses = h_pmasses;

    // Set the tree AABB.
    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = -1.f;
    max_x = max_y = max_z = 1.f;
    float4 bot = make_float4(min_x, min_y, min_z, 0.f);
    float4 top = make_float4(max_x, max_y, max_z, 0.f);

    // Generate morton keys based on particles' positions.
    thrust::device_vector<grace::uinteger32> d_keys(N);
    thrust::device_vector<grace::uinteger32> d_keys_2(N);
    grace::morton_keys(d_spheres_xyzr, d_keys, bot, top);
    thrust::copy(d_keys.begin(), d_keys.end(), d_keys_2.begin());

    // Sort particle positions and smoothing lengths by their keys.
    thrust::sort_by_key(d_keys_2.begin(), d_keys_2.end(),
                        d_spheres_xyzr.begin());
    d_keys_2.clear(); d_keys_2.shrink_to_fit();
    // Sort other properties by the same keys.
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_pmasses.begin());

    // Build the tree hierarchy from the keys.
    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);
    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);

    // Generate the rays, emitted from box side (X, Y, min_z-max_r) and of
    // length (max_z + max_r) - (min_z - max_r).
    // Since we want to fully include all particles, the ray (ox, oy) limits
    // are set by [min_x-max_r, max_x+max_r] and [min_y-max_r, max_y+max_r].
    // Rays at the edges are likely to have no hits!
    float max_r = 0.2f;
    float span_x = 2*max_r + max_x - min_x;
    float span_y = 2*max_r + max_y - min_y;
    float span_z = 2*max_r + max_z - min_z;
    // N_rays_side - 1 so if we start at min_x-max_r, we end at max_x+max_r.
    // Same for y.
    float spacer_x = span_x / (N_rays_side-1);
    float spacer_y = span_y / (N_rays_side-1);
    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<grace::uinteger32> h_keys(N_rays);
    int i, j;
    float ox, oy;
    for (i=0, ox=min_x-max_r; i<N_rays_side; ox+=spacer_x, i++)
    {
        for (j=0, oy=min_y-max_r; j<N_rays_side; oy+=spacer_y, j++)
        {
            // All rays point in +ve z direction.
            h_rays[i*N_rays_side +j].dx = 0.0f;
            h_rays[i*N_rays_side +j].dy = 0.0f;
            h_rays[i*N_rays_side +j].dz = 1.0f;

            h_rays[i*N_rays_side +j].ox = ox;
            h_rays[i*N_rays_side +j].oy = oy;
            h_rays[i*N_rays_side +j].oz = min_z - max_r;

            h_rays[i*N_rays_side +j].length = span_z;
            h_rays[i*N_rays_side +j].dclass = 7;

            // Since all rays are PPP, base key on origin instead.
            // morton_key(float, float, float) requires floats in (0, 1).
            h_keys[i*N_rays_side + j] = grace::morton_key((ox-min_x)/span_x,
                                                          (oy-min_y)/span_y,
                                                          0.0f);
        }
    }

    // Sort rays by Morton key and trace z-integrated pseudo-mass,
    // which is equal to one for every particle.
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<float> d_traced_mass(N_rays);
    // Copy tabulated kernel integrals to device.
    thrust::device_vector<float> d_b_integrals(grace::kernel_integral_table,
                                               grace::kernel_integral_table+51);
    grace::gpu::trace_property<<<28, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        d_rays.size(),
        thrust::raw_pointer_cast(d_traced_mass.data()),
        thrust::raw_pointer_cast(d_nodes.lrpe.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        d_nodes.lrpe.size(),
        thrust::raw_pointer_cast(d_spheres_xyzr.data()),
        thrust::raw_pointer_cast(d_pmasses.data()),
        thrust::raw_pointer_cast(d_b_integrals.data()));
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );

    // Calculate the sum of all integrals.  (~ Integrating over x and y).
    float integrated_total = thrust::reduce(d_traced_mass.begin(),
                                            d_traced_mass.end(),
                                            0.0f,
                                            thrust::plus<float>());

    // Should multiply each ray by its effective area when integrating.
    // Since all areas are equal, we can do it after summing.
    integrated_total *= (spacer_x * spacer_y);

    std::cout << "Number of rays:         " << N_rays << std::endl;
    std::cout << "Number of particles:    " << N << std::endl;
    std::cout << "Total integral:         " << integrated_total << std::endl;
    std::cout << "Integral / N_particles: " << integrated_total / N
              << std::endl;
    std::cout << std::endl;
} // End device code.  Call all thrust destructors etc. before cudaDeviceReset().

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
