// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "nodes.h"
#include "ray.h"
#include "util/extrema.cuh"
#include "kernels/trace_sph.cuh"
#include "helper/cuda_timer.cuh"
#include "helper/rays.cuh"
#include "helper/read_gadget.cuh"
#include "helper/tree.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);

    size_t N_rays = 512 * 512; // = 262,144
    int max_per_leaf = 32;
    std::string fname = "../data/gadget/0128/Data_025";
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
        device_ID = (unsigned int) std::strtol(argv[5], NULL, 10);
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

    std::cout << "Gadget file:            " << fname << std::endl;
    // Vector is resized in read_gadget().
    thrust::device_vector<float4> d_spheres;
    read_gadget(fname, d_spheres);
    const size_t N = d_spheres.size();

    std::cout << "Number of particles:    " << N << std::endl
              << "Number of rays:         " << N_rays << std::endl
              << "Max particles per leaf: " << max_per_leaf << std::endl
              << "Number of iterations:   " << N_iter << std::endl
              << "Running on device:      " << device_ID
                                            << " (" << deviceProp.name << ")"
                                            << std::endl
              << std::endl;


    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::Tree d_tree(N, max_per_leaf);

    // build_tree can compute the x/y/z limits for us, but we compute them
    // explicitly as we also need them for othogonal_rays_z.
    float4 mins, maxs;
    grace::min_vec4(d_spheres, &mins);
    grace::max_vec4(d_spheres, &maxs);

    build_tree(d_spheres, mins, maxs, d_tree);
    orthogonal_rays_z(N_per_side, mins, maxs, d_rays);


    CUDATimer timer;
    double total = 0.0;
    thrust::device_vector<float> d_integrals(N_rays);
    for (int i = -1; i < N_iter; ++i)
    {
        timer.start();
        grace::trace_cumulative_sph(d_rays,
                                    d_spheres,
                                    d_tree,
                                    d_integrals);
        if (i >= 0) total += timer.elapsed();

        // Must be done in-loop for cuMemGetInfo to return relevant results.
        if (i == 0) {
            // Temporary memory used in tree construction is impossible to
            // (straightforwardly) compute, so below we only include the
            // 'permanently' allocated memory.
            size_t trace_bytes = 0;
            trace_bytes += d_spheres.size() * sizeof(float4);
            trace_bytes += d_tree.nodes.size() * sizeof(int4);
            trace_bytes += d_tree.leaves.size() * sizeof(int4);
            trace_bytes += d_rays.size() * sizeof(grace::Ray);
            trace_bytes += d_integrals.size() * sizeof(float);
            trace_bytes += grace::N_table * sizeof(double); // Integral lookup.

            std::cout << "Memory allocated for cumulative trace kernel:    "
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

    std::cout << "Time for cumulative density tracing: " << std::setw(8)
              << total / N_iter << " ms" << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler (nvprof/nvvp) trace.
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
