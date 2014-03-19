#include <cmath>
#include <sstream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

    std::ifstream file;
    std::string fname = "Data_025";
    std::cout << "Reading in data from Gadget file " << fname << "..."
              << std::endl;

    // Read in gas data from Gadget-2 file.
    // Arrays are resized in read_gadget_gas().
    thrust::host_vector<float4> h_spheres_xyzr(1);
    thrust::host_vector<float> h_masses(1);
    thrust::host_vector<float> h_rho(1);

    file.open(fname.c_str(), std::ios::binary);
    grace::read_gadget_gas(file, h_spheres_xyzr, h_masses, h_rho);
    file.close();

    size_t N = h_spheres_xyzr.size();
    std::cout << "Will trace " << N_rays << " rays through " << N
              << " particles..." << std::endl;
    std::cout << std::endl;

    // Masses unused.
    h_masses.clear(); h_masses.shrink_to_fit();


// Device code.
{
    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;
    thrust::device_vector<float> d_rho = h_rho;

    // Set the tree AABB.
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
    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<float> d_sorted(N);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

    thrust::gather(d_indices.begin(), d_indices.end(),
                   d_rho.begin(), d_sorted.begin());
    d_rho = d_sorted;

    // Clear temporary storage.
    d_sorted.clear(); d_sorted.shrink_to_fit();
    d_indices.clear(); d_indices.shrink_to_fit();

    // Build the tree hierarchy from the keys.
    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);
    grace::build_nodes(d_nodes, d_leaves, d_keys);
    // Keys no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);

        // Generate the rays (emitted from box centre (.5, .5, .5) of length 2).
    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<float> h_dxs(N_rays);
    thrust::host_vector<float> h_dys(N_rays);
    thrust::host_vector<float> h_dzs(N_rays);
    thrust::host_vector<grace::uinteger32> h_keys(N_rays);
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
    // Ensure rays end (well) outside box.
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

        // morton_key(float, float, float) requires floats in (0, 1).
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

    // Sort rays by Morton key and trace for per-ray hit couynts.
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<unsigned int> d_hit_counts(N_rays);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    grace::gpu::trace_hitcount<<<28, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        d_rays.size(),
        thrust::raw_pointer_cast(d_hit_counts.data()),
        thrust::raw_pointer_cast(d_nodes.left.data()),
        thrust::raw_pointer_cast(d_nodes.right.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        d_nodes.left.size(),
        thrust::raw_pointer_cast(d_spheres_xyzr.data()));
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    int max_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                  0, thrust::maximum<unsigned int>());
    int min_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                  N+1, thrust::minimum<unsigned int>());
    float mean_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                     0, thrust::plus<unsigned int>())
                                    / float(N_rays);
    std::cout << "Time for hit-count tracing kernel: " << elapsed << " ms"
              << std::endl;


    thrust::device_vector<float> d_traced_rho(N_rays);
    // Copy tabulated kernel integrals to device.
    thrust::device_vector<float> d_b_integrals(grace::kernel_integral_table,
                                               grace::kernel_integral_table+51);
    // Trace and integrate through smoothing kernels, accumulating density.
    cudaEventRecord(start);
    grace::gpu::trace_property<<<28, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        d_rays.size(),
        thrust::raw_pointer_cast(d_traced_rho.data()),
        thrust::raw_pointer_cast(d_nodes.left.data()),
        thrust::raw_pointer_cast(d_nodes.right.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        d_nodes.left.size(),
        thrust::raw_pointer_cast(d_spheres_xyzr.data()),
        thrust::raw_pointer_cast(d_rho.data()),
        thrust::raw_pointer_cast(d_b_integrals.data()));
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    // Find min, max of output vector for 'plotting'.
    float max_rho = thrust::reduce(d_traced_rho.begin(), d_traced_rho.end(),
                                   0.0f, thrust::maximum<float>());
    float min_rho = thrust::reduce(d_traced_rho.begin(), d_traced_rho.end(),
                                   1E20, thrust::minimum<float>());
    float mean_rho = thrust::reduce(d_traced_rho.begin(), d_traced_rho.end(),
                                    0.0f, thrust::plus<float>()) / d_traced_rho.size();
    std::cout << "Time for acummulating integrating tracing kernel: "
              << elapsed << " ms" << std::endl;


    // Allocate output array based on per-ray hit counts, and calculate
    // individual ray offsets into this array.
    // int last_ray_hits = d_hit_counts[N_rays-1];
    // thrust::exclusive_scan(d_hit_counts.begin(), d_hit_counts.end(),
    //                        d_hit_counts.begin());
    // thrust::device_vector<float> d_trace_output(d_hit_counts[N_rays-1]+
    //                                             last_ray_hits);
    // thrust::device_vector<float> d_trace_distances(d_trace_output.size());

    // // Trace and integrate through smoothing kernels, accumulating density.
    // cudaEventRecord(start);
    // grace::gpu::trace<<<28, TRACE_THREADS_PER_BLOCK>>>(
    //     thrust::raw_pointer_cast(d_rays.data()),
    //     d_rays.size(),
    //     thrust::raw_pointer_cast(d_trace_output.data()),
    //     thrust::raw_pointer_cast(d_trace_distances.data()),
    //     thrust::raw_pointer_cast(d_hit_counts.data()),
    //     thrust::raw_pointer_cast(d_nodes.data()),
    //     thrust::raw_pointer_cast(d_leaves.data()),
    //     d_nodes.size(),
    //     thrust::raw_pointer_cast(d_spheres_xyzr.data()),
    //     thrust::raw_pointer_cast(d_rho.data()),
    //     thrust::raw_pointer_cast(d_b_integrals.data()));
    // CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    // CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsed, start, stop);
    // std::cout << "Time for per-intersection integrating kernel: " << elapsed
    //           << " ms" << std::endl;

    // // Sort output arrays based on hit distances.
    // thrust::host_vector<int> h_hit_counts = d_hit_counts;
    // double t = 0.0;
    // for (int i=0; i<N_rays_side; i++) {
    //     int r_start = h_hit_counts[i];
    //     int r_end;
    //     if (i == N_rays-1)
    //         r_end = h_hit_counts[i] + last_ray_hits - 1;
    //     else
    //         r_end = h_hit_counts[i+1] - 1;
    //     cudaEventRecord(start);
    //     thrust::sort_by_key(d_trace_distances.begin()+r_start,
    //                         d_trace_distances.begin()+r_end,
    //                         d_trace_output.begin()+r_start);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&elapsed, start, stop);
    //     t += elapsed;
    // }
    // std::cout << "Time for per-intersection sorting loop: " << t << " ms"
    //           << std::endl;
    std::cout << std::endl;

    std::cout << "Number of rays:       " << N_rays << std::endl;
    std::cout << "Number of particles:  " << N << std::endl;
    std::cout << "Mean hits:            " << mean_hits << std::endl;
    std::cout << "Max hits:             " << max_hits << std::endl;
    std::cout << "Min hits:             " << min_hits << std::endl;
    std::cout << "Mean output           " << mean_rho << std::endl;
    std::cout << "Max output:           " << max_rho << std::endl;
    std::cout << "Min output:           " << min_rho << std::endl;
    std::cout << std::endl;

} // End device code.  Call all thrust destructors etc. before cudaDeviceReset().

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
