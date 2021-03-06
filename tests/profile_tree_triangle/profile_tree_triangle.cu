// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "triangle.cuh"

#include "grace/types.h"
#include "grace/cuda/nodes.h"
#include "grace/cuda/kernels/aabb.cuh"
#include "grace/cuda/kernels/albvh.cuh"
#include "grace/cuda/kernels/morton.cuh"
#include "grace/cuda/util/extrema.cuh"
#include "grace/generic/functors/albvh.h"
#include "helper/cuda_timer.cuh"
#include "helper/read_ply.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);

    int max_per_leaf = 32;
    int N_iter = 100;
    std::string fname = "../data/ply/dragon_recon/dragon_vrip.ply";
    unsigned int device_ID = 0;

    if (argc > 1) {
        max_per_leaf = (int)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_iter = (int)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        fname = std::string(argv[3]);
    }
    if (argc > 4) {
        device_ID = (unsigned int)std::strtol(argv[4], NULL, 10);
    }

    /* Output run parameters and device properties to console. */

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

{   // Device code. To ensure that cudaDeviceReset() does not fail, all Thrust
    // vectors should be allocated within this block. (The closing } brace
    // causes them to be freed before we call cudaDeviceReset(); if device
    // vectors are not freed, cudaDeviceReset() will throw.)

    std::cout << "Input geometry file:     " << fname << std::endl;
    // Vector is resized in read_triangles().
    std::vector<PLYTriangle> ply_tris;
    thrust::host_vector<Triangle> h_tris;
    read_triangles(fname, ply_tris);
    h_tris = ply_tris;
    const size_t N = h_tris.size();

    std::cout << "Number of primitives:    " << N << std::endl
              << "Max primitives per leaf: " << max_per_leaf << std::endl
              << "Number of iterations:    " << N_iter << std::endl
              << "Running on device:       " << device_ID
                                             << " (" << deviceProp.name << ")"
                                             << std::endl
              << std::endl;

    CUDATimer timer;
    double t_all, t_bounds, t_morton, t_sort, t_deltas, t_leaves, t_leaf_deltas, t_nodes;
    t_all = t_bounds = t_morton = t_sort = t_deltas = t_leaves = t_leaf_deltas = t_nodes = 0.0;
    for (int i = -1; i < N_iter; ++i)
    {
        timer.start();

        thrust::device_vector<Triangle> d_tris = h_tris;
        thrust::device_vector<float3> d_centroids(N);
        thrust::device_vector<grace::uinteger32> d_keys(N);
        thrust::device_vector<grace::uinteger32> d_deltas(N + 1);
        grace::Tree d_tree(N, max_per_leaf);
        thrust::device_vector<int2> d_tmp_nodes(N - 1);
        // Don't include above memory allocations in t_morton.
        timer.split();

        grace::AABB::compute_centroids(
            thrust::raw_pointer_cast(d_tris.data()),
            N,
            thrust::raw_pointer_cast(d_centroids.data()),
            TriangleCentroid());
        float3 bots, tops;
        grace::min_vec3(thrust::raw_pointer_cast(d_centroids.data()), N, &bots);
        grace::max_vec3(thrust::raw_pointer_cast(d_centroids.data()), N, &tops);
        if (i >= 0) t_bounds += timer.split();

        grace::morton_keys(d_tris, bots, tops, d_keys, TriangleCentroid());
        if (i >= 0) t_morton += timer.split();

        thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                            d_tris.begin());
        if (i >= 0) t_sort += timer.split();

        grace::compute_deltas(d_keys, d_deltas, grace::DeltaXOR());
        if (i >= 0) t_deltas += timer.split();

        grace::ALBVH::build_leaves(
            d_tmp_nodes,
            d_tree.leaves,
            d_tree.max_per_leaf,
            thrust::raw_pointer_cast(d_deltas.data()),
            thrust::less<grace::uinteger32>());
        grace::ALBVH::remove_empty_leaves(d_tree);
        if (i >= 0) t_leaves += timer.split();

        const size_t n_new_leaves = d_tree.leaves.size();
        thrust::device_vector<grace::uinteger32> d_new_deltas(n_new_leaves + 1);
        // Don't include above memory allocation in t_leaf_deltas.
        timer.split();

        grace::ALBVH::copy_leaf_deltas(
            d_tree.leaves,
            thrust::raw_pointer_cast(d_deltas.data()),
            thrust::raw_pointer_cast(d_new_deltas.data()));
        if (i >= 0) t_leaf_deltas += timer.split();

        grace::ALBVH::build_nodes(
            d_tree,
            thrust::raw_pointer_cast(d_tris.data()),
            thrust::raw_pointer_cast(d_new_deltas.data()),
            thrust::less<grace::uinteger32>(),
            TriangleAABB());
        if (i >= 0) t_nodes += timer.split();

        if (i >= 0) t_all += timer.elapsed();
    }

    std::cout << "Time to compute bounds:            " << std::setw(7)
              << t_bounds/N_iter << " ms." << std::endl
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

} // End device code.

    // Exit cleanly to ensure full profiler (nvprof/nvvp) trace.
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
