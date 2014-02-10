#include <cmath>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "utils.cuh"
#include "../types.h"
#include "../nodes.h"
#include "../ray.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build_kernels.cuh"
#include "../kernels/bintree_trace.cuh"

int main(int argc, char* argv[])
{
    typedef grace::Vector3<float> Vector3f;

    unsigned int N = 1000000;
    unsigned int N_rays = 100000;
    // Do we save the input and output data?
    bool save_data = false;

    if (argc > 3) {
        if (strcmp("save", argv[3]) == 0)
            save_data = true;
    }
    if (argc > 2) {
        N_rays = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }

    std::cout << "Generating " << N << " random points and " << N_rays
              << " random rays." << std::endl;
    if (save_data)
        std::cout << "Will save sphere, ray and hit data." << std::endl;
    std::cout << std::endl;
{

    // Generate N random positions and radii, i.e. 4N random floats in [0,1).
    thrust::device_vector<float> d_x_centres(N);
    thrust::device_vector<float> d_y_centres(N);
    thrust::device_vector<float> d_z_centres(N);
    thrust::device_vector<float> d_radii(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_x_centres.begin(),
                      random_float_functor(0u) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_y_centres.begin(),
                      random_float_functor(1u) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_z_centres.begin(),
                      random_float_functor(2u) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_radii.begin(),
                      random_float_functor(0.0f, 0.1f) );

    // Set the centre-containing AABBs.
    Vector3f bottom(0., 0., 0.);
    Vector3f top(1., 1., 1.);

    // Sort the positions by their keys and save the sorted keys.
    thrust::device_vector<UInteger32> d_keys(N);
    grace::morton_keys(d_x_centres, d_y_centres, d_z_centres,
                       d_keys,
                       bottom, top);

    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<float> d_tmp(N);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

    thrust::gather(d_indices.begin(),
                   d_indices.end(),
                   d_x_centres.begin(),
                   d_tmp.begin());
    d_x_centres = d_tmp;

    thrust::gather(d_indices.begin(),
                   d_indices.end(),
                   d_y_centres.begin(),
                   d_tmp.begin());
    d_y_centres = d_tmp;

    thrust::gather(d_indices.begin(),
                   d_indices.end(),
                   d_z_centres.begin(),
                   d_tmp.begin());
    d_z_centres = d_tmp;

    thrust::gather(d_indices.begin(),
                   d_indices.end(),
                   d_radii.begin(),
                   d_tmp.begin());
    d_radii = d_tmp;
    // Clear temporary storage.
    d_tmp.clear();
    d_tmp.shrink_to_fit();
    d_indices.clear();
    d_indices.shrink_to_fit();

    // Build the tree hierarchy from the keys.
    thrust::device_vector<grace::Node> d_nodes(N-1);
    thrust::device_vector<grace::Leaf> d_leaves(N);
    grace::build_nodes(d_nodes, d_leaves, d_keys);
    // Keys no longer needed.
    d_keys.clear();
    d_keys.shrink_to_fit();
    grace::find_AABBs(d_nodes, d_leaves,
                      d_x_centres, d_y_centres, d_z_centres, d_radii);

    // Generate the rays (emitted from box centre (.5, .5, .5) of length 2).
    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<float> h_dxs(N_rays);
    thrust::host_vector<float> h_dys(N_rays);
    thrust::host_vector<float> h_dzs(N_rays);
    thrust::host_vector<UInteger32> h_keys(N_rays);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N_rays),
                      h_dxs.begin(),
                      random_float_functor(0u, -1.0f, 1.0f) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N_rays),
                      h_dys.begin(),
                      random_float_functor(1u, -1.0f, 1.0f) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N_rays),
                      h_dzs.begin(),
                      random_float_functor(2u, -1.0f, 1.0f) );
    for (int i=0; i<N_rays; i++) {
        float N_dir = sqrt(h_dxs[i]*h_dxs[i] +
                           h_dys[i]*h_dys[i] +
                           h_dzs[i]*h_dzs[i]);
        h_rays[i].dx = h_dxs[i] / N_dir;
        h_rays[i].dy = h_dys[i] / N_dir;
        h_rays[i].dz = h_dzs[i] / N_dir;
        h_rays[i].ox = h_rays[i].oy = h_rays[i].oz = 0.5f;
        h_rays[i].length = 2;
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
        thrust::raw_pointer_cast(d_nodes.data()),
        thrust::raw_pointer_cast(d_leaves.data()),
        d_nodes.size(),
        thrust::raw_pointer_cast(d_x_centres.data()),
        thrust::raw_pointer_cast(d_y_centres.data()),
        thrust::raw_pointer_cast(d_z_centres.data()),
        thrust::raw_pointer_cast(d_radii.data()));
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    int max_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                  0u, thrust::maximum<unsigned int>());
    int min_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                  N, thrust::minimum<unsigned int>());
    float mean_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                     0u, thrust::plus<unsigned int>())
                                    / float(N_rays);
    std::cout << "Time for hit-count tracing kernel: " << elapsed
              << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Number of rays:       " << N_rays << std::endl;
    std::cout << "Number of particles:  " << N << std::endl;
    std::cout << "Mean hits:            " << mean_hits << std::endl;
    std::cout << "Max hits:             " << max_hits << std::endl;
    std::cout << "Min hits:             " << min_hits << std::endl;

    if (save_data)
    {
        std::ofstream outfile;

        outfile.setf(std::ios::fixed, std::ios::floatfield);
        outfile.precision(9);
        outfile.width(11);
        outfile.fill('0');

        thrust::host_vector<float> h_x_centres = d_x_centres;
        thrust::host_vector<float> h_y_centres = d_y_centres;
        thrust::host_vector<float> h_z_centres = d_z_centres;
        thrust::host_vector<float> h_radii = d_radii;
        outfile.open("indata/spheredata.txt");
        for (int i=0; i<N; i++) {
            outfile << h_x_centres[i] << " " << h_y_centres[i] << " "
                    << h_z_centres[i] << " " << h_radii[i] << std::endl;
        }
        outfile.close();

        outfile.open("indata/raydata.txt");
        for (int i=0; i<N_rays; i++) {
            outfile << h_rays[i].dx << " " << h_rays[i].dy << " "
                    << h_rays[i].dz << " " << h_rays[i].ox << " "
                    << h_rays[i].oy << " " << h_rays[i].oz << " "
                    << h_rays[i].length << std::endl;
        }
        outfile.close();

        thrust::host_vector<float> h_hit_counts = d_hit_counts;
        outfile.open("outdata/hitdata.txt");
        for (int i=0; i<N_rays; i++) {
            outfile << h_hit_counts[i] << std::endl;
        }
        outfile.close();
    }

}
    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
