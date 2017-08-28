// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/build_sph.cuh"
#include "grace/cuda/bvh.cuh"
#include "grace/cuda/detail/kernels/albvh.cuh"
#include "grace/generic/functors/albvh.h"
#include "grace/aabb.h"
#include "grace/types.h"
#include "grace/sphere.h"
#include "helper/cuda_timer.cuh"
#include "helper/random.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

typedef grace::Sphere<float> SphereType;

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);

    int max_per_leaf = 32;
    int N_iter = 100;
    int log2N_min = 20;
    int log2N_max = 23; // Should run on most devices.
    unsigned int device_ID = 0;

    if (argc > 1) {
        max_per_leaf = (int)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_iter = (int)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        log2N_max = (int)std::strtol(argv[3], NULL, 10);
        log2N_max = min(28, max(5, log2N_max));
        if (log2N_max < log2N_min)
            log2N_min = log2N_max;
    }
    if (argc > 4) {
        log2N_max = (int)std::strtol(argv[4], NULL, 10);
        // Keep levels in [5, 28].
        log2N_max = min(28, max(5, log2N_max));
        log2N_min = (int)std::strtol(argv[3], NULL, 10);
        log2N_min = min(28, max(5, log2N_min));
    }
    if (argc > 5) {
        device_ID = (unsigned int)std::strtol(argv[5], NULL, 10);
    }

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    std::cout << "Max particles per leaf:   " << max_per_leaf << std::endl
              << "Iterations per tree:      " << N_iter << std::endl
              << "Starting log2(N_points):  " << log2N_min << std::endl
              << "Finishing log2(N_points): " << log2N_max << std::endl
              << "Running on device:        " << device_ID
                                            << " (" << deviceProp.name << ")"
                                            << std::endl
              << std::endl;

{   // Device code. To ensure that cudaDeviceReset() does not fail, all Thrust
    // vectors should be allocated within this block. (The closing } brace
    // causes them to be freed before we call cudaDeviceReset(); if device
    // vectors are not freed, cudaDeviceReset() will throw.)

    for (int p = log2N_min; p <= log2N_max; ++p)
    {
        size_t N = 1u << p;

        SphereType high = SphereType(1.0f, 1.0f, 1.0f, 0.1f);
        SphereType low = SphereType(0.0f, 0.0f, 0.0f, 0.0f);

        thrust::host_vector<SphereType> h_spheres(N);
        thrust::transform(thrust::counting_iterator<unsigned int>(0),
                          thrust::counting_iterator<unsigned int>(N),
                          h_spheres.begin(),
                          random_sphere_functor<SphereType>(low, high));

        CUDATimer timer;
        double t_all, t_morton, t_sort, t_deltas, t_leaves, t_leaf_deltas, t_nodes;
        t_all = t_morton = t_sort = t_deltas = t_leaves = t_leaf_deltas = t_nodes = 0.0;
        for (int i = -1; i < N_iter; ++i)
        {
            timer.start();

            thrust::device_vector<SphereType> d_spheres = h_spheres;
            thrust::device_vector<grace::uinteger32> d_keys(N);
            thrust::device_vector<float> d_deltas(N + 1);
            grace::CudaBvh d_bvh(N, max_per_leaf);
            grace::detail::Bvh_ref<grace::CudaBvh> bvh_ref(d_bvh);
            thrust::device_vector<int2> d_tmp_nodes(N - 1);
            thrust::device_vector<grace::detail::CudaBvhLeaf> d_tmp_leaves(N);
            // Don't include above memory allocations in t_morton.
            timer.split();

            grace::morton_keys_sph(d_spheres,
                                   grace::AABB<float>(low.center(),
                                                      high.center()),
                                   d_keys);
            if (i >= 0) t_morton += timer.split();

            thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                                d_spheres.begin());
            if (i >= 0) t_sort += timer.split();

            grace::euclidean_deltas_sph(d_spheres, d_deltas);
            if (i >= 0) t_deltas += timer.split();

            grace::detail::build_leaves(
                d_tmp_nodes,
                d_tmp_leaves,
                d_bvh.max_per_leaf(),
                thrust::raw_pointer_cast(d_deltas.data()),
                thrust::less<float>());
            grace::detail::remove_empty_leaves(d_tmp_leaves);
            if (i >= 0) t_leaves += timer.split();

            bvh_ref.leaves() = d_tmp_leaves;
            const size_t n_new_leaves = d_bvh.num_leaves();
            const size_t n_new_nodes = n_new_leaves - 1;
            bvh_ref.nodes().resize(n_new_nodes);

            thrust::device_vector<float> d_new_deltas(n_new_leaves + 1);
            // Don't include above memory allocation in t_leaf_deltas.
            timer.split();

            grace::detail::copy_leaf_deltas(
                bvh_ref.leaves(),
                thrust::raw_pointer_cast(d_deltas.data()),
                thrust::raw_pointer_cast(d_new_deltas.data()));
            if (i >= 0) t_leaf_deltas += timer.split();

            grace::detail::build_nodes(
                d_bvh,
                thrust::raw_pointer_cast(d_spheres.data()),
                thrust::raw_pointer_cast(d_new_deltas.data()),
                thrust::less<float>(),
                grace::AABBSphere());
            if (i >= 0) t_nodes += timer.split();

            // Record the total time spent in the loop.
            if (i >= 0) t_all += timer.elapsed();
        }

        std::cout << "Number of particles:               " << N << std::endl
                  << "Time for Morton key generation:    " << std::setw(7)
                  << t_morton/N_iter << " ms." << std::endl
                  << "Time for sort-by-key:              " << std::setw(7)
                  << t_sort/N_iter << " ms." << std::endl
                  << "Time for computing deltas:         " << std::setw(7)
                  << t_deltas/N_iter << " ms." << std::endl
                  << "Time for building leaves:          " << std::setw(7)
                  << t_leaves/N_iter << " ms." << std::endl
                  << "Time for computing leaf deltas:    " << std::setw(7)
                  << t_leaf_deltas/N_iter << " ms." << std::endl
                  << "Time for building nodes:           " << std::setw(7)
                  << t_nodes/N_iter << " ms." << std::endl
                  << "Time for total (inc. memory ops):  " << std::setw(7)
                  << t_all/N_iter << " ms." << std::endl
                  << std::endl;
    }
} // End device code.

    // Exit cleanly to ensure full profiler (nvprof/nvvp) trace.
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
