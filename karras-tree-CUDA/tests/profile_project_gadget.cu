#include <cmath>
#include <sstream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/sort.cuh"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);


    /* Initialize run parameters. */

    unsigned int device_ID = 0;
    unsigned int N_rays = 512*512;
    unsigned int max_per_leaf = 32;
    unsigned int N_iter = 10;

    if (argc > 1) {
        device_ID = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        max_per_leaf = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        N_iter = (unsigned int) std::strtol(argv[4], NULL, 10);
    }

    unsigned int N_rays_side = floor(pow(N_rays, 0.500001));


    /* Read in Gadget file. */

    std::ifstream file;
    std::string fname = "Data_025";

    // Arrays are resized in read_gadget_gas().
    thrust::host_vector<float4> h_spheres_xyzr(1);
    thrust::host_vector<unsigned int> h_gadget_IDs(1);
    thrust::host_vector<float> h_masses(1);
    thrust::host_vector<float> h_rho(1);

    file.open(fname.c_str(), std::ios::binary);
    grace::read_gadget_gas(file, h_spheres_xyzr,
                                 h_gadget_IDs,
                                 h_masses,
                                 h_rho);
    file.close();

    size_t N = h_spheres_xyzr.size();

    // Gadget IDs and masses unused.
    h_gadget_IDs.clear(); h_gadget_IDs.shrink_to_fit();
    h_masses.clear(); h_masses.shrink_to_fit();


    /* Output run parameters and device properties to console. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    std::cout << "Device " << device_ID
                    << ":                   " << deviceProp.name << std::endl;
    std::cout << "TRACE_THREADS_PER_BLOCK:    "
              << grace::TRACE_THREADS_PER_BLOCK << std::endl;
    std::cout << "MAX_BLOCKS:                 "
              << grace::MAX_BLOCKS << std::endl;
    std::cout << "Gadget data file:           " << fname << std::endl;
    std::cout << "Number of gas particles:    " << N << std::endl;
    std::cout << "Number of rays:             " << N_rays << std::endl;
    std::cout << "Max particles per leaf:     " << max_per_leaf << std::endl;
    std::cout << "Number of iterations:       " << N_iter << std::endl;
    std::cout << std::endl << std::endl;


{ // Device code.


    /* Build the tree. */

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
    thrust::device_vector<float> d_rho = h_rho;
    thrust::device_vector<unsigned int> d_keys(N);

    grace::morton_keys(d_keys, d_spheres_xyzr);
    grace::sort_by_key(d_keys, d_spheres_xyzr, d_rho);

    grace::Tree d_tree(N);

    grace::build_tree(d_tree, d_keys, max_per_leaf);
    grace::compact_tree(d_tree);
    grace::find_AABBs(d_tree, d_spheres_xyzr);


    /* Generate the rays, all emitted in +z direction from a box side. */

    // Rays emitted from box side (x, y, min_z - max_r) and of length
    // (max_z + max_r) - (min_z - max_r).  For simplicity, the ray (ox, oy)
    // limits are determined only by the particle min(x), min(y) and max(x),
    // max(y) limits and smoothing lengths are ignored.  This ensures that rays
    // at the edge will hit something!
    float min_x, max_x;
    grace::min_max_x(&min_x, &max_x, d_spheres_xyzr);

    float min_y, max_y;
    grace::min_max_y(&min_y, &max_y, d_spheres_xyzr);

    float min_z, max_z;
    grace::min_max_z(&min_z, &max_z, d_spheres_xyzr);

    float min_r, max_r;
    grace::min_max_w(&min_r, &max_r, d_spheres_xyzr);

    float span_x = max_x - min_x;
    float span_y = max_y - min_y;
    float span_z = 2*max_r + max_z - min_z;
    float spacer_x = span_x / (N_rays_side-1);
    float spacer_y = span_y / (N_rays_side-1);

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<unsigned int> h_ray_keys(N_rays);

    int i, j;
    float ox, oy;
    for (i=0, ox=min_x; i<N_rays_side; ox+=spacer_x, i++)
    {
        for (j=0, oy=min_y; j<N_rays_side; oy+=spacer_y, j++)
        {
            // All rays point in +ve z direction.
            h_rays[i*N_rays_side + j].dx = 0.0f;
            h_rays[i*N_rays_side + j].dy = 0.0f;
            h_rays[i*N_rays_side + j].dz = 1.0f;

            h_rays[i*N_rays_side + j].ox = ox;
            h_rays[i*N_rays_side + j].oy = oy;
            h_rays[i*N_rays_side + j].oz = min_z - max_r;

            h_rays[i*N_rays_side + j].length = span_z;

            // Since all rays are PPP, base key on origin instead.
            // Floats must be in (0, 1) for morton_key().
            h_ray_keys[i*N_rays_side + j] = grace::morton_key((ox-min_x)/span_x,
                                                              (oy-min_y)/span_y,
                                                              0.0f);
        }
    }


    /* Profile the tracing performance by tracing rays through the simulation
     * data and accumulating density.  Repeat N_iter times.
     */

    cudaEvent_t part_start, part_stop;
    cudaEvent_t tot_start, tot_stop;
    float elapsed;
    double sort_tot, trace_tot, all_tot;
    cudaEventCreate(&part_start);
    cudaEventCreate(&part_stop);
    cudaEventCreate(&tot_start);
    cudaEventCreate(&tot_stop);

    for (int i=0; i<N_iter; i++) {
        cudaEventRecord(tot_start);

        thrust::device_vector<grace::Ray> d_rays = h_rays;
        thrust::device_vector<unsigned int> d_ray_keys = h_ray_keys;

        cudaEventRecord(part_start);
        thrust::sort_by_key(d_ray_keys.begin(), d_ray_keys.end(),
                            d_rays.begin());
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        sort_tot += elapsed;

        thrust::device_vector<float> d_traced_rho(N_rays);

        cudaEventRecord(part_start);
        grace::trace_property<float>(d_rays,
                                     d_traced_rho,
                                     d_tree,
                                     d_spheres_xyzr,
                                     max_per_leaf,
                                     d_rho);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        trace_tot += elapsed;

        cudaEventRecord(tot_stop);
        cudaEventSynchronize(tot_stop);
        cudaEventElapsedTime(&elapsed, tot_start, tot_stop);
        all_tot += elapsed;

        // Must be done in-loop for cuMemGetInfo to return relevant results.
        if (i == 0) {
            float trace_bytes = 0.0;
            float unused_bytes = 0.0;
            trace_bytes += d_rays.size() * sizeof(grace::Ray);
            trace_bytes += d_traced_rho.size() * sizeof(float);
            trace_bytes += d_tree.nodes.size() * sizeof(int4);
            trace_bytes += d_tree.leaves.size() * sizeof(int4);
            trace_bytes += d_spheres_xyzr.size() * sizeof(float4);
            trace_bytes += d_rho.size() * sizeof(float);
            trace_bytes += grace::N_table * sizeof(float); // Integral lookup.

            unused_bytes += d_keys.size() * sizeof(unsigned int);
            unused_bytes += d_tree.levels.size() * sizeof(unsigned int);
            unused_bytes += d_ray_keys.size() * sizeof(unsigned int);

            std::cout << "Total memory for property trace kernel:    "
                      << trace_bytes / (1024.*1024.*1024.) << " GiB"
                      << std::endl;
            std::cout << "Allocated memory not used in trace kernel: "
                      << unused_bytes / (1024.*1024.*1024.) << " GiB"
                      << std::endl;

            size_t avail, total;
            cuMemGetInfo(&avail, &total);
            std::cout << "Free memory:  " << avail / (1024.*1024.*1024.)
                      << " GiB" << std::endl;
            std::cout << "Total memory: " << total / (1024.*1024.*1024.)
                      << " GiB" << std::endl;
            std::cout << std::endl;
        }
    }

    std::cout << "Time for ray sort-by-key:             ";
    std::cout.width(8);
    std::cout << sort_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for cummulative density tracing: ";
    std::cout.width(8);
    std::cout << trace_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for total (inc. memory ops):     ";
    std::cout.width(8);
    std::cout << all_tot / N_iter << " ms" << std::endl;

    std::cout << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
