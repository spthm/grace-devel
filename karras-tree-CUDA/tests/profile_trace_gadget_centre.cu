// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include <cmath>
#include <sstream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include "../kernel_config.h"
#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/gen_rays.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/sort.cuh"


int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);


    /* Initialize run parameters. */

    unsigned int device_ID = 0;
    unsigned int N_rays = 4532 * 32; // = 145024 ~ max on an M2090 for data set.
    unsigned int max_per_leaf = 32;
    unsigned int N_iter = 2;

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
    thrust::device_vector<grace::uinteger32> d_keys(N);
    thrust::device_vector<float> d_deltas(N+1);

    grace::morton_keys(d_keys, d_spheres_xyzr);
    grace::sort_by_key(d_keys, d_spheres_xyzr, d_rho);

    grace::Tree d_tree(N, max_per_leaf);

    grace::compute_deltas(d_spheres_xyzr, d_deltas);
    grace::build_tree(d_tree, d_spheres_xyzr, d_deltas, d_spheres_xyzr);


    /* Compute information needed for ray generation; rays are emitted from the
     * box centre and of sufficient length to be terminated outside the box.
     */

    // Assume x, y and z spatial extents are similar.
    float min, max;
    grace::min_max_x(&min, &max, d_spheres_xyzr);
    float x_centre = (max + min) / 2.;
    float y_centre = x_centre;
    float z_centre = x_centre;
    float length = 2 * (max - min) * sqrt(3);


    /* Profile the tracing performance by tracing rays through the simulation
     * data and i) accumulating density and ii) saving column densities
     * and distances to each intersected particle.  Repeat N_iter times.
     */

    cudaEvent_t part_start, part_stop;
    cudaEvent_t tot_start, tot_stop;
    float elapsed;
    double gen_ray_tot, sort_rho_dists_tot;
    double trace_rho_tot, trace_full_tot, trace_hit_tot;
    double all_tot;
    cudaEventCreate(&part_start);
    cudaEventCreate(&part_stop);
    cudaEventCreate(&tot_start);
    cudaEventCreate(&tot_stop);

    for (int i=0; i<N_iter; i++) {
        cudaEventRecord(tot_start);

        thrust::device_vector<grace::Ray> d_rays(N_rays);
        cudaEventRecord(part_start);
        grace::uniform_random_rays(d_rays,
                                   x_centre, y_centre, z_centre, length);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        gen_ray_tot += elapsed;

        // The cumulated, integrated density for each ray.
        thrust::device_vector<float> d_traced_rho(N_rays);

        cudaEventRecord(part_start);
        grace::trace_property<float>(d_rays,
                                     d_traced_rho,
                                     d_tree,
                                     d_spheres_xyzr,
                                     d_rho);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        trace_rho_tot += elapsed;


        /* Full trace. */

        // Indices of particles for all ray-particle intersections.
        thrust::device_vector<unsigned int> d_hit_indices;
        // Distances, from the ray origin, to all ray-particle intersections.
        thrust::device_vector<float> d_hit_distances;
        // Offsets into the above vector where each ray's data starts.
        thrust::device_vector<int> d_ray_offsets(N_rays);

        cudaEventRecord(part_start);
        grace::trace<float>(d_rays,
                            d_traced_rho,
                            d_ray_offsets,
                            d_hit_indices,
                            d_hit_distances,
                            d_tree,
                            d_spheres_xyzr,
                            d_rho); // For RT, we'd pass ~number counts.
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        trace_full_tot += elapsed;

        cudaEventRecord(part_start);
        grace::sort_by_distance(d_hit_distances,
                                d_ray_offsets,
                                d_hit_indices,
                                d_traced_rho);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        sort_rho_dists_tot += elapsed;

        /* End of full trace. */

        // Also profile the pure hit-count tracing.  Useful for optimizing the
        // pure tracing performance.
        cudaEventRecord(part_start);
        grace::trace_hitcounts(d_rays,
                               d_ray_offsets,
                               d_tree,
                               d_spheres_xyzr);
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        trace_hit_tot += elapsed;

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
            trace_bytes += d_ray_offsets.size() * sizeof(float);
            trace_bytes += d_hit_indices.size() * sizeof(unsigned int);
            trace_bytes += d_tree.nodes.size() * sizeof(int4);
            trace_bytes += d_tree.leaves.size() * sizeof(int4);
            trace_bytes += d_spheres_xyzr.size() * sizeof(float4);
            trace_bytes += d_rho.size() * sizeof(float);
            trace_bytes += grace::N_table * sizeof(float); // Integral lookup.
            trace_bytes += d_hit_distances.size() * sizeof(float);

            unused_bytes += d_keys.size() * sizeof(grace::uinteger32);
            unused_bytes += d_deltas.size() * sizeof(float);
            // Ray keys, used when generating rays.
            unused_bytes += d_rays.size() * sizeof(unsigned int);

            std::cout << "Total hits: " << d_hit_indices.size() << std::endl;
            std::cout << std::endl;
            std::cout << "Total memory for full trace kernel and sort: "
                      << trace_bytes / (1024.*1024.*1024.) << " GiB"
                      << std::endl;
            std::cout << "Allocated memory not used in trace kernel:   "
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

    std::cout << "Time for generating and sorting rays:   ";
    std::cout.width(8);
    std::cout << gen_ray_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for hit count tracing:             ";
    std::cout.width(8);
    std::cout << trace_hit_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for cumulative density tracing:    ";
    std::cout.width(8);
    std::cout << trace_rho_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for full tracing:                  ";
    std::cout.width(8);
    std::cout << trace_full_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for sort-by-distance:              ";
    std::cout.width(8);
    std::cout << sort_rho_dists_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for total (inc. memory ops):       ";
    std::cout.width(8);
    std::cout << all_tot / N_iter << " ms" << std::endl;

    std::cout << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
