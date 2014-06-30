#include <cmath>
#include <sstream>
#include <fstream>

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
    unsigned int max_per_leaf = 10;

    if (argc > 1)
        N_rays = (unsigned int) std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        max_per_leaf = (unsigned int) std::strtol(argv[2], NULL, 10);

    unsigned int N_rays_side = floor(pow(N_rays, 0.500001));


    /* Read in Gadget file. */

    std::ifstream file;
    std::string fname = "Data_025";
    std::cout << "Reading in data from Gadget file " << fname << "..."
              << std::endl;

    // Arrays are resized in read_gadget_gas().
    thrust::host_vector<float4> h_spheres_xyzr(1);
    thrust::host_vector<unsigned int> h_gadget_IDs(1);
    thrust::host_vector<float> h_pmasses(1);
    thrust::host_vector<float> h_rho(1);

    file.open(fname.c_str(), std::ios::binary);
    grace::read_gadget_gas(file, h_spheres_xyzr,
                           h_gadget_IDs,
                           h_pmasses,
                           h_rho);
    file.close();

    size_t N = h_spheres_xyzr.size();
    std::cout << "Will trace " << N_rays << " rays through " << N
              << " particles, with up to " << max_per_leaf << " particle(s)"
              << std::endl
              << "per leaf..." << std::endl;
    std::cout << std::endl;

    // Gadget IDs and densities unused.
    h_gadget_IDs.clear(); h_gadget_IDs.shrink_to_fit();
    h_rho.clear(); h_rho.shrink_to_fit();

    // Set all (pseudo)masses equal to 1/N so the total sum is one.
    for (int i=0; i<N; i++) {
        h_pmasses[i] = 1.0f/N;
    }


{ // Device code.


    /* Build the tree. */

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
    thrust::device_vector<float> d_pmasses = h_pmasses;

    // Calculate limits here explicity since we need them later (i.e. do not
    // get morton_keys() to do it for us).
    float min_x, max_x;
    grace::min_max_x(&min_x, &max_x, d_spheres_xyzr);

    float min_y, max_y;
    grace::min_max_y(&min_y, &max_y, d_spheres_xyzr);

    float min_z, max_z;
    grace::min_max_z(&min_z, &max_z, d_spheres_xyzr);

    float min_r, max_r;
    grace::min_max_w(&min_r, &max_r, d_spheres_xyzr);

    float3 top = make_float3(max_x, max_y, max_z);
    float3 bot = make_float3(min_x, min_y, min_z);

    // One set of keys for sorting spheres, one for sorting an arbitrary
    // number of other properties.
    thrust::device_vector<unsigned int> d_keys(N);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
    grace::sort_by_key(d_keys, d_spheres_xyzr, d_pmasses);

    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys, max_per_leaf);
    grace::compact_nodes(d_nodes, d_leaves);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);

    // Keys no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();


    /* Generate the rays, all emitted in +z direction from a box side. */

    // Rays emitted from box side (x, y, min_z - max_r) and of length
    // (max_z + max_r) - (min_z - max_r).  Since we want to fully include all
    // particles, the ray (ox, oy) limits are set by:
    // [min_x - max_r, max_x + max_r] and [min_y - max_r, max_y + max_r].
    // Rays at the edges are likely to have no hits!
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


    /* Trace and accumulate pseudomass through the simulation data. */

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;

    thrust::device_vector<float> d_traced_pmass(N_rays);

    grace::trace_property<float>(d_rays,
                                 d_traced_pmass,
                                 d_nodes,
                                 d_leaves,
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
