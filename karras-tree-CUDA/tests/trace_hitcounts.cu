#include <cmath>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "utils.cuh"
#include "../nodes.h"
#include "../ray.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"

int main(int argc, char* argv[]) {

    std::ofstream outfile;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(9);

    /* Initialize run parameters. */

    unsigned int N = 1000000;
    unsigned int N_rays = 100000;
    bool save_data = false;

    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        if (strcmp("save", argv[3]) == 0)
            save_data = true;
    }

    std::cout << "Generating " << N << " random points and " << N_rays
              << " random rays." << std::endl;
    if (save_data)
        std::cout << "Will save all data." << std::endl;
    std::cout << std::endl;

{ // Device code.

    /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

    thrust::device_vector<float4> d_spheres_xyzr(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f) );


    /* Build the tree from the random data. */

    thrust::device_vector<unsigned int> d_keys(N);
    float4 bot = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 top = make_float4(1.f, 1.f, 1.f, 0.f);
    grace::morton_keys(d_spheres_xyzr, d_keys, bot, top);

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres_xyzr.begin());

    grace::Nodes d_nodes(N-1);
    grace::Leaves d_leaves(N);
    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_spheres_xyzr);

    // Keys no longer needed.  Free space.
    d_keys.clear();
    d_keys.shrink_to_fit();


    /* Generate the rays (emitted from box centre and of length 2). */

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<float> h_dxs(N_rays);
    thrust::host_vector<float> h_dys(N_rays);
    thrust::host_vector<float> h_dzs(N_rays);
    thrust::host_vector<unsigned int> h_keys(N_rays);

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

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;


    /* Trace for per-ray hit counts. */

    thrust::device_vector<unsigned int> d_hit_counts(N_rays);
    grace::gpu::trace_hitcounts_kernel<<<28, TRACE_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        d_rays.size(),
        thrust::raw_pointer_cast(d_hit_counts.data()),
        thrust::raw_pointer_cast(d_nodes.hierarchy.data()),
        thrust::raw_pointer_cast(d_nodes.AABB.data()),
        d_nodes.hierarchy.size(),
        thrust::raw_pointer_cast(d_spheres_xyzr.data()));
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );


    /* Output simple hit-count statistics. */

    int max_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                  0u, thrust::maximum<unsigned int>());
    int min_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                  N, thrust::minimum<unsigned int>());
    float mean_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(),
                                     0u, thrust::plus<unsigned int>())
                                    / float(N_rays);

    std::cout << "Number of rays:       " << N_rays << std::endl;
    std::cout << "Number of particles:  " << N << std::endl;
    std::cout << "Mean hits:            " << mean_hits << std::endl;
    std::cout << "Max hits:             " << max_hits << std::endl;
    std::cout << "Min hits:             " << min_hits << std::endl;


    /* Save sphere, ray and hit count data. */

    if (save_data)
    {
        thrust::host_vector<float4> h_spheres_xyzr = d_spheres_xyzr;
        outfile.open("indata/spheredata.txt");
        for (int i=0; i<N; i++) {
            outfile << h_spheres_xyzr[i].x << " " << h_spheres_xyzr[i].y << " "
                    << h_spheres_xyzr[i].z << " " << h_spheres_xyzr[i].w
                    << std::endl;
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

} // End device code.

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
