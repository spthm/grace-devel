#include <cmath>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/sort.cuh"

int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    unsigned int N_rays = 250000;

    if (argc > 1)
        N_rays = (unsigned int) std::strtol(argv[1], NULL, 10);

    unsigned int N_rays_side = floor(pow(N_rays, 0.500001));


    /* Generate two spheres (tree does not work for N < 2 objects). */

    unsigned int N = 2;
    thrust::host_vector<float4> h_spheres_xyzr(N);

    float radius = 0.2f;
    h_spheres_xyzr[0].x = -0.5f;
    h_spheres_xyzr[0].y = 0.0f;
    h_spheres_xyzr[0].z = 0.0f;
    h_spheres_xyzr[0].w = radius;

    h_spheres_xyzr[1].x = +0.5f;
    h_spheres_xyzr[1].y = 0.0f;
    h_spheres_xyzr[1].z = 0.0f;
    h_spheres_xyzr[1].w = radius;

    // Set both (pseudo)masses equal to .5, so the total sum is one.
    thrust::host_vector<float> h_pmasses(N);
    for (int i=0; i<N; i++) {
        h_pmasses[i] = 0.5f;
    }

    std::cout << "Will trace " << N_rays << " rays through " << N
              << " particles." << std::endl;
    std::cout << std::endl;

{ // Device code.


    /* Build the tree. */

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
    thrust::device_vector<float> d_pmasses = h_pmasses;

    float max_x, max_y, max_z, min_x, min_y, min_z;
    max_x = max_y = max_z = 1.f;
    min_x = min_y = min_z = -1.f;
    float3 top = make_float3(max_x, max_y, max_z);
    float3 bot = make_float3(min_x, min_y, min_z);

    thrust::device_vector<unsigned int> d_keys(N);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
    grace::sort_by_key(d_keys, d_spheres_xyzr, d_pmasses);

    grace::Tree d_tree(N);

    grace::build_tree(d_tree, d_keys);
    grace::find_AABBs(d_tree, d_spheres_xyzr);

    /* Generate the rays, all emitted in +z direction from a box side. */

    // Rays emitted from box side (x, y, min_z - max_r) and of length
    // (max_z + max_r) - (min_z - max_r).  Since we want to fully include all
    // particles, the ray (ox, oy) limits are set by:
    // [min_x - max_r, max_x + max_r] and [min_y - max_r, max_y + max_r].
    // Rays at the edges are likely to have no hits!
    float max_r = radius;
    float span_x = 2*max_r + max_x - min_x;
    float span_y = 2*max_r + max_y - min_y;
    float span_z = 2*max_r + max_z - min_z;
    float spacer_x = span_x / (N_rays_side-1);
    float spacer_y = span_y / (N_rays_side-1);

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<unsigned int> h_keys(N_rays);

    int i, j;
    float ox, oy;
    for (i=0, ox=min_x-max_r; i<N_rays_side; ox+=spacer_x, i++)
    {
        for (j=0, oy=min_y-max_r; j<N_rays_side; oy+=spacer_y, j++)
        {
            h_rays[i*N_rays_side +j].dx = 0.0f;
            h_rays[i*N_rays_side +j].dy = 0.0f;
            h_rays[i*N_rays_side +j].dz = 1.0f;

            h_rays[i*N_rays_side +j].ox = ox;
            h_rays[i*N_rays_side +j].oy = oy;
            h_rays[i*N_rays_side +j].oz = min_z - max_r;

            h_rays[i*N_rays_side +j].length = span_z;
            h_rays[i*N_rays_side +j].dclass = 7;

            // Since all rays are PPP, base key on origin instead.
            // Floats must be in (0, 1) for morton_key().
            h_keys[i*N_rays_side + j] = grace::morton_key((ox-min_x)/span_x,
                                                          (oy-min_y)/span_y,
                                                          0.0f);
        }
    }


    /* Trace and accumulate pseudomass through the two spheres. */

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;

    thrust::device_vector<float> d_traced_pmass(N_rays);

    grace::trace_property<float>(d_rays,
                                 d_traced_pmass,
                                 d_tree,
                                 d_spheres_xyzr,
                                 d_pmasses);

    // ~ Integrate over x and y.
    float integrated_total = thrust::reduce(d_traced_pmass.begin(),
                                            d_traced_pmass.end(),
                                            0.0f,
                                            thrust::plus<float>());
    // Multiply by the pixel area to complete the x-y integration.
    integrated_total *= (spacer_x * spacer_y);

    std::cout << "Number of rays:               " << N_rays << std::endl;
    std::cout << "Number of particles:          " << N << std::endl;
    std::cout << "Volume integrated pseudomass: " << integrated_total
              << std::endl;
    std::cout << std::endl;
} // End device code.

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
