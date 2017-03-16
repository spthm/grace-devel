// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/generate_rays.cuh"
#include "grace/cuda/prngstates.cuh"
#include "grace/cuda/trace_sph.cuh"
#include "grace/cuda/util/extrema.cuh"
#include "grace/ray.h"
#include "grace/sphere.h"
#include "grace/vector.h"
#include "helper/cuda_timer.cuh"
#include "helper/read_gadget.cuh"
#include "helper/tree.cuh"

#include <thrust/device_vector.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

typedef grace::Sphere<float> SphereType;

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);

    size_t N_rays = 1200 * 32; // = 38,400; should run on most devices.
    int max_per_leaf = 32;
    std::string fname = "../data/gadget/0128/Data_025";
    int N_iter = 2;
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

{   // Device code. To ensure that cudaDeviceReset() does not fail, all Thrust
    // vectors should be allocated within this block. (The closing } brace
    // causes them to be freed before we call cudaDeviceReset(); if device
    // vectors are not freed, cudaDeviceReset() will throw.)

    std::cout << "Gadget file:            " << fname << std::endl;
    // Vector is resized in read_gadget().
    thrust::device_vector<SphereType> d_spheres;
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


    grace::PrngStates rng_states;
    grace::Tree d_tree(N, max_per_leaf);
    build_tree(d_spheres, d_tree);

    // Ray origin is the box centre; all rays will exit the box.
    // Assume x, y and z spatial extents are similar.
    float min, max, length;
    grace::Vector<3, float> origin;
    grace::min_max_x(d_spheres, &min, &max);
    origin.x = origin.y = origin.z = (max + min) / 2.;
    length = 2 * (max - min);

    CUDATimer timer;
    double t_genray, t_sort, t_cum, t_trace, t_hit, t_all;
    t_genray = t_sort = t_cum = t_trace = t_hit = t_all = 0.0;
    for (int i = -1; i < N_iter; ++i)
    {
        timer.start();

        thrust::device_vector<grace::Ray> d_rays(N_rays);
        thrust::device_vector<int> d_ray_offsets(N_rays);
        // Resized in grace::trace_sph():
        thrust::device_vector<float> d_integrals(N_rays);
        thrust::device_vector<int> d_indices;
        thrust::device_vector<float> d_distances;
        // Don't include above memory allocations in t_genray.
        timer.split();

        grace::uniform_random_rays(origin, length, rng_states, d_rays);
        if (i >= 0) t_genray += timer.split();

        grace::trace_cumulative_sph(d_rays,
                                    d_spheres,
                                    d_tree,
                                    d_integrals);
        if (i >= 0) t_cum += timer.split();

        grace::trace_sph(d_rays,
                         d_spheres,
                         d_tree,
                         d_ray_offsets,
                         d_indices,
                         d_integrals,
                         d_distances);
        if (i >= 0) t_trace += timer.split();

        grace::sort_by_distance(d_distances,
                                d_ray_offsets,
                                d_indices,
                                d_integrals);
        if (i >= 0) t_sort += timer.split();

        // Profiling the pure hit-count tracing is useful for optimizing the
        // traversal algorithm.
        grace::trace_hitcounts_sph(d_rays,
                                   d_spheres,
                                   d_tree,
                                   d_ray_offsets);
        if (i >= 0) t_hit += timer.split();

        if (i >= 0) t_all += timer.elapsed();

        // Must be done in-loop for cuMemGetInfo to return relevant results.
        if (i == 0) {
            // Temporary memory used in tree construction is impossible to
            // (straightforwardly) compute, so below we only include the
            // 'permanently' allocated memory.
            float trace_bytes = 0.0;
            trace_bytes += d_spheres.size() * sizeof(SphereType);
            trace_bytes += d_tree.leaves.size() * sizeof(int4);
            trace_bytes += d_tree.nodes.size() * sizeof(int4);
            trace_bytes += d_rays.size() * sizeof(grace::Ray);
            trace_bytes += d_ray_offsets.size() * sizeof(int);
            trace_bytes += d_integrals.size() * sizeof(float);
            trace_bytes += d_indices.size() * sizeof(int);
            trace_bytes += d_distances.size() * sizeof(float);
            trace_bytes += grace::N_table * sizeof(double); // Integral lookup.

            std::cout << "Total hits: " << d_indices.size() << std::endl
                      << std::endl
                      << "Total memory for full trace kernel and sort: "
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

    std::cout << "Time for generating and sorting rays:   " << std::setw(8)
              << t_genray / N_iter << " ms" << std::endl
              << "Time for hit count tracing:             " << std::setw(8)
              << t_hit / N_iter << " ms" << std::endl
              << "Time for cumulative density tracing:    " << std::setw(8)
              << t_cum / N_iter << " ms" << std::endl
              << "Time for full tracing:                  " << std::setw(8)
              << t_trace / N_iter << " ms" << std::endl
              << "Time for sort-by-distance:              " << std::setw(8)
              << t_sort / N_iter << " ms" << std::endl
              << "Time for total (inc. memory ops):       " << std::setw(8)
              << t_all / N_iter << " ms" << std::endl
              << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler (nvprof/nvvp) trace.
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
