// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "triangle.cuh"
#include "tris_tree.cuh"
#include "tris_trace.cuh"

#include "grace/cuda/nodes.h"
#include "grace/cuda/generate_rays.cuh"
#include "grace/aabb.h"
#include "grace/ray.h"
#include "grace/vector.h"
#include "helper/cuda_timer.cuh"

#include <thrust/device_vector.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);

    size_t N_rays = 512 * 512; // = 262,144
    int max_per_leaf = 32;
    std::string fname = "../data/ply/dragon_recon/dragon_vrip.ply";
    int N_iter = 10;
    unsigned int device_ID = 0;

    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        max_per_leaf = (int)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        fname = std::string(argv[3]);
    }
    if (argc > 4) {
        N_iter = (int)std::strtol(argv[4], NULL, 10);
    }
    if (argc > 5) {
        device_ID = (unsigned int)std::strtol(argv[5], NULL, 10);
    }

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    size_t N_per_side = std::floor(std::pow(N_rays, 0.500001));
    // N_rays must be a multiple of 32.
    N_per_side = ((N_per_side + 32 - 1) / 32) * 32;
    N_rays = N_per_side * N_per_side;

{   // Device code. To ensure that cudaDeviceReset() does not fail, all Thrust
    // vectors should be allocated within this block. (The closing } brace
    // causes them to be freed before we call cudaDeviceReset(); if device
    // vectors are not freed, cudaDeviceReset() will throw.)

    std::cout << "Input geometry file:     " << fname << std::endl;
    // Vector is resized in read_triangles().
    std::vector<PLYTriangle> ply_tris;
    thrust::device_vector<Triangle> d_tris;
    read_triangles(fname, ply_tris);
    d_tris = ply_tris;
    const size_t N = d_tris.size();

    std::cout << "Number of primitives:    " << N << std::endl
              << "Number of rays:          " << N_rays << std::endl
              << "Max primitives per leaf: " << max_per_leaf << std::endl
              << "Number of iterations:    " << N_iter << std::endl
              << "Running on device:       " << device_ID
                                             << " (" << deviceProp.name << ")"
                                             << std::endl
              << std::endl;

    grace::AABB<float> aabb;
    grace::Tree d_tree(N, max_per_leaf);
    build_tree_tris(d_tris, d_tree, &aabb);

    std::vector<grace::Vector<3, float> > camera_positions;
    grace::Vector<3, float> look_at, view_up;
    float FOVy_radians, ray_length;
    float FOVy_degrees = 50.f;
    setup_cameras(aabb, FOVy_degrees, N_per_side, N_per_side,
                  camera_positions, &look_at, &view_up,
                  &FOVy_radians, &ray_length);

    CUDATimer timer;
    double t_genray, t_closest, t_all;
    t_genray = t_closest = t_all = 0.0;
    for (int i = -1; i < N_iter; ++i)
    {
        for (int j = 0; j < camera_positions.size(); ++j)
        {
            timer.start();

            thrust::device_vector<grace::Ray> d_rays(N_rays);
            thrust::device_vector<int> d_closest_tri_idx(N_rays);
            // Don't include above memory allocations in t_genray.
            timer.split();

            pinhole_camera_rays(camera_positions[j], look_at, view_up,
                                FOVy_radians, ray_length,
                                N_per_side, N_per_side,
                                d_rays);
            if (i >= 0) t_genray += timer.split();

            trace_closest_tri(
                d_rays,
                d_tris,
                d_tree,
                d_closest_tri_idx);
            if (i >= 0) t_closest += timer.split();

            if (i >= 0) t_all += timer.elapsed();

            // Must be done in-loop for cuMemGetInfo to return relevant results.
            if (i == 0 && j == 0) {
                // Temporary memory used in tree construction is impossible to
                // (straightforwardly) compute, so below we only include the
                // 'permanently' allocated memory.
                float trace_bytes = 0.0;
                trace_bytes += d_tris.size() * sizeof(Triangle);
                trace_bytes += d_tree.leaves.size() * sizeof(int4);
                trace_bytes += d_tree.nodes.size() * sizeof(int4);
                trace_bytes += d_rays.size() * sizeof(grace::Ray);
                trace_bytes += d_closest_tri_idx.size() * sizeof(int);

                std::cout << std::endl
                          << "Total memory for closest-triangle traversal: "
                          << trace_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB"
                          << std::endl;

                size_t avail, total;
                cuMemGetInfo(&avail, &total);
                std::cout << "Free memory:  " << avail / (1024.0 * 1024.0 * 1024.0)
                          << " GiB" << std::endl
                          << "Total memory: " << total / (1024.0 * 1024.0 * 1024.0)
                          << " GiB" << std::endl
                          << std::endl;
            }
        }
    }

    size_t N_trials = N_iter * camera_positions.size();
    std::cout << "Time for generating rays:             " << std::setw(8)
              << t_genray / N_trials << " ms" << std::endl
              << "Time for closest-triangle traversal:  " << std::setw(8)
              << t_closest / N_trials << " ms" << std::endl
              << "Time for total (inc. memory ops):     " << std::setw(8)
              << t_all / N_trials << " ms" << std::endl
              << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler (nvprof/nvvp) trace.
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
