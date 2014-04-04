#include <cmath>
#include <sstream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
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

    cudaDeviceProp deviceProp;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);


    /* Initialize run parameters. */

    unsigned int device_ID = 0;
    unsigned int N_rays = 250000;
    unsigned int N_iter = 10;

    if (argc > 1) {
        device_ID = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        N_iter = (unsigned int) std::strtol(argv[3], NULL, 10);
    }

    unsigned int N_rays_side = floor(pow(N_rays, 0.500001));


    /* Read in Gadget file. */

    std::ifstream file;
    std::string fname = "Data_025";

    // Arrays are resized in read_gadget_gas().
    thrust::host_vector<float4> h_spheres_xyzr(1);
    thrust::host_vector<float> h_masses(1);
    thrust::host_vector<float> h_rho(1);

    file.open(fname.c_str(), std::ios::binary);
    grace::read_gadget_gas(file, h_spheres_xyzr, h_masses, h_rho);
    file.close();

    size_t N = h_spheres_xyzr.size();

    // Masses unused.
    h_masses.clear(); h_masses.shrink_to_fit();


    /* Output run parameters and device properties to console. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    std::cout << "Device " << device_ID
                    << ":                   " << deviceProp.name << std::endl;
    std::cout << "TRACE_THREADS_PER_BLOCK:    " << TRACE_THREADS_PER_BLOCK
            << std::endl;
    std::cout << "MAX_BLOCKS:                 " << MAX_BLOCKS << std::endl;
    std::cout << "Gadget data file:           " << fname << std::endl;
    std::cout << "Number of gas particles:    " << N << std::endl;
    std::cout << "Number of rays:             " << N_rays << std::endl;
    std::cout << "Number of iterations:       " << N_iter << std::endl;
    std::cout << std::endl << std::endl;


{ // Device code.


    /* Build the tree. */

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
    thrust::device_vector<float> d_rho = h_rho;

    float min_x, max_x;
    grace::min_max_x(&min_x, &max_x, d_spheres_xyzr);

    float min_y, max_y;
    grace::min_max_y(&min_y, &max_y, d_spheres_xyzr);

    float min_z, max_z;
    grace::min_max_z(&min_z, &max_z, d_spheres_xyzr);

    float min_r, max_r;
    grace::min_max_w(&min_r, &max_r, d_spheres_xyzr);

    float4 bot = make_float4(min_x, min_y, min_z, 0.f);
    float4 top = make_float4(max_x, max_y, max_z, 0.f);

    // One set of keys for sorting spheres, one for sorting an arbitrary
    // number of other properties.
    thrust::device_vector<unsigned int> d_keys(N);
    thrust::device_vector<unsigned int> d_keys_2(N);

    grace::morton_keys(d_spheres_xyzr, d_keys, bot, top);
    thrust::copy(d_keys.begin(), d_keys.end(), d_keys_2.begin());

    thrust::sort_by_key(d_keys_2.begin(), d_keys_2.end(),
                        d_spheres_xyzr.begin());
    d_keys_2.clear(); d_keys_2.shrink_to_fit();

    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<float> d_sorted(N);

    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());
    thrust::gather(d_indices.begin(), d_indices.end(),
                   d_rho.begin(), d_sorted.begin());

    d_rho = d_sorted;

    // Working arrays no longer needed.
    d_sorted.clear(); d_sorted.shrink_to_fit();
    d_indices.clear(); d_indices.shrink_to_fit();

    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);


    /* Generate the rays, all emitted in +z direction from a box side. */

    // Rays emitted from box side (x, y, min_z - max_r) and of length
    // (max_z + max_r) - (min_z - max_r).  For simplicity, the ray (ox, oy)
    // limits are determined only by the particle min(x, y) / max(x, y) limits
    // and smoothing lengths are ignored.  This ensures that rays at the edge
    // will hit something!
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
            h_rays[i*N_rays_side + j].dclass = 7;

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
        thrust::sort_by_key(d_ray_keys.begin(), d_ray_keys.end(), d_rays.begin());
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        sort_tot += elapsed;

        thrust::device_vector<float> d_traced_rho(N_rays);
        grace::KernelIntegrals<float> lookup;
        thrust::device_vector<float> d_b_integrals(&lookup.table[0],
                                                   &lookup.table[50]);

        cudaEventRecord(part_start);
        grace::gpu::trace_property_kernel<<<28, TRACE_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_rays.data()),
            d_rays.size(),
            thrust::raw_pointer_cast(d_traced_rho.data()),
            thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
            thrust::raw_pointer_cast(d_nodes.AABB.data()),
            d_nodes.hierarchy.size(),
            thrust::raw_pointer_cast(d_spheres_xyzr.data()),
            thrust::raw_pointer_cast(d_rho.data()),
            thrust::raw_pointer_cast(d_b_integrals.data()));
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
            trace_bytes += d_nodes.hierarchy.size() * sizeof(int4);
            trace_bytes += d_nodes.AABB.size() * sizeof(grace::Box);
            trace_bytes += d_spheres_xyzr.size() * sizeof(float4);
            trace_bytes += d_rho.size() * sizeof(float);
            trace_bytes += d_b_integrals.size() * sizeof(float);

            unused_bytes += d_keys.size() * sizeof(unsigned int);
            unused_bytes += d_nodes.level.size() * sizeof(unsigned int);
            unused_bytes += d_leaves.parent.size() * sizeof(int);
            unused_bytes += d_leaves.AABB.size() * sizeof(grace::Box);
            unused_bytes += d_ray_keys.size() * sizeof(unsigned int);

            std::cout << "Total memory for property trace kernel:        "
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
