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
    thrust::device_vector<grace::uinteger32> d_keys(N);
    thrust::device_vector<grace::uinteger32> d_keys_2(N);

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

    // // Keys no longer needed.
    // d_keys.clear(); d_keys.shrink_to_fit();
    // // Levels unused.
    // d_nodes.level.clear(); d_nodes.level.shrink_to_fit();
    // // Leaves unused.
    // d_leaves.parent.clear(); d_leaves.parent.shrink_to_fit();
    // d_leaves.AABB.clear(); d_leaves.AABB.shrink_to_fit();


    /* Generate the rays, emitted emitted from box centre (.5, .5, .5) and of
     * sufficient length to be terminated outwith the box.
     */

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<grace::uinteger32> h_keys(N_rays);

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

    float x_centre = (max_x+min_x) / 2.;
    float y_centre = (max_y+min_y) / 2.;
    float z_centre = (max_z+min_z) / 2.;

    float length = sqrt((max_x-min_x)*(max_x-min_x) +
                        (max_y-min_y)*(max_y-min_y) +
                        (max_z-min_z)*(max_z-min_z));

    for (int i=0; i<N_rays; i++) {
        float N_dir = sqrt(h_dxs[i]*h_dxs[i] +
                           h_dys[i]*h_dys[i] +
                           h_dzs[i]*h_dzs[i]);

        h_rays[i].dx = h_dxs[i] / N_dir;
        h_rays[i].dy = h_dys[i] / N_dir;
        h_rays[i].dz = h_dzs[i] / N_dir;

        h_rays[i].ox = x_centre;
        h_rays[i].oy = y_centre;
        h_rays[i].oz = z_centre;

        h_rays[i].length = length;

        h_rays[i].dclass = 0;
        if (h_dxs[i] >= 0)
            h_rays[i].dclass += 1;
        if (h_dys[i] >= 0)
            h_rays[i].dclass += 2;
        if (h_dzs[i] >= 0)
            h_rays[i].dclass += 4;

        // Floats must be in (0, 1) for morton_key().
        h_keys[i] = grace::morton_key((h_rays[i].dx+1)/2.f,
                                      (h_rays[i].dy+1)/2.f,
                                      (h_rays[i].dz+1)/2.f);
    }
    h_dxs.clear();
    h_dxs.shrink_to_fit();
    h_dys.clear();
    h_dys.shrink_to_fit();
    h_dzs.clear();
    h_dxs.shrink_to_fit();


    /* Profile the tracing performance by tracing rays through the simulation
     * data and i) accumulating density and ii) saving column densities
     * and distances to each intersected particle.  Repeat N_iter times.
     */

    cudaEvent_t part_start, part_stop;
    cudaEvent_t tot_start, tot_stop;
    float elapsed;
    double sort_ray_tot, sort_dists_tot;
    double trace_rho_tot, trace_full_tot;
    double all_tot;
    cudaEventCreate(&part_start);
    cudaEventCreate(&part_stop);
    cudaEventCreate(&tot_start);
    cudaEventCreate(&tot_stop);

    for (int i=0; i<N_iter; i++) {
        cudaEventRecord(tot_start);

        thrust::device_vector<grace::Ray> d_rays = h_rays;
        thrust::device_vector<unsigned int> d_keys = h_keys;

        cudaEventRecord(part_start);
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_rays.begin());
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        sort_ray_tot += elapsed;

        // d_keys.clear(); d_keys.shrink_to_fit();

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
        trace_rho_tot += elapsed;

        thrust::device_vector<unsigned int> d_hit_offsets(N_rays);


        /* Full trace. */

        cudaEventRecord(part_start);

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
        int last_ray_hits = d_hit_offsets[N_rays-1];
        thrust::exclusive_scan(d_hit_offsets.begin(), d_hit_offsets.end(),
                               d_hit_offsets.begin());

        d_traced_rho.resize(d_hit_offsets[N_rays-1] + last_ray_hits);
        thrust::device_vector<float> d_trace_dists(d_traced_rho.size());

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

        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        cudaEventElapsedTime(&elapsed, part_start, part_stop);
        trace_full_tot += elapsed;

        /* End of full trace. */


        thrust::host_vector<int> h_hit_offsets = d_hit_offsets;

        for (int ray_i=0; ray_i<N_rays; ray_i++) {
            int ray_start = h_hit_offsets[ray_i];
            int ray_end;

            if (ray_i == N_rays-1)
                ray_end = h_hit_offsets[ray_i] + last_ray_hits - 1;
            else
                ray_end = h_hit_offsets[ray_i+1] - 1;

            cudaEventRecord(part_start);
            thrust::sort_by_key(d_trace_dists.begin() + ray_start,
                                d_trace_dists.begin() + ray_end,
                                d_traced_rho.begin() + ray_start);
            cudaEventRecord(part_stop);
            cudaEventSynchronize(part_stop);
            cudaEventElapsedTime(&elapsed, part_start, part_stop);
            sort_dists_tot += elapsed;
        }

        cudaEventRecord(tot_stop);
        cudaEventSynchronize(tot_stop);
        cudaEventElapsedTime(&elapsed, tot_start, tot_stop);
        all_tot += elapsed;

        if (i == 0) {
            float total_bytes = 0.0;
            total_bytes += d_rays.size() * sizeof(grace::Ray);
            total_bytes += d_traced_rho.size() * sizeof(float);
            total_bytes += d_trace_dists.size() * sizeof(float);
            total_bytes += d_hit_offsets.size() * sizeof(unsigned int);
            total_bytes += d_nodes.hierarchy.size() * sizeof(int4);
            total_bytes += d_nodes.AABB.size() * sizeof(grace::Box);
            total_bytes += d_spheres_xyzr.size() * sizeof(float);
            total_bytes += d_rho.size() * sizeof(float);
            total_bytes += d_b_integrals.size() * sizeof(float);

            std::cout << "Total memory for full trace kernel: "
                      << total_bytes / (1024.*1024.*1024.) << " GB"
                      << std::endl;

            size_t avail, total;
            cuMemGetInfo(&avail, &total);
            std::cout << "Free memory:  " << avail / (1024.*1024.*1024.)
                      << " GB" << std::endl;
            std::cout << "Total memory: " << total / (1024.*1024.*1024.)
                      << " GB" << std::endl;
            std::cout << std::endl;
        }
    }

    std::cout << "Time for ray sort-by-key:             ";
    std::cout.width(8);
    std::cout << sort_ray_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for cummulative density tracing: ";
    std::cout.width(8);
    std::cout << trace_rho_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for full tracing:                ";
    std::cout.width(8);
    std::cout << trace_full_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for ray hits sort-by-distance:   ";
    std::cout.width(8);
    std::cout << sort_dists_tot / N_iter << " ms" << std::endl;

    std::cout << "Time for total (inc. memory ops):     ";
    std::cout.width(8);
    std::cout << all_tot / N_iter << " ms" << std::endl;

    std::cout << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
