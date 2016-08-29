// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/generate_rays.cuh"
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

    int max_per_leaf = 32;
    std::string fname = "../data/gadget/0128/Data_025";
    int N_iter = 2;
    unsigned int device_ID = 0;

    if (argc > 1) {
        max_per_leaf = (int)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        fname = std::string(argv[2]);
    }
    if (argc > 3) {
        N_iter = (int)std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        device_ID = (unsigned int)std::strtol(argv[4], NULL, 10);
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
              << "Number of rays:         " << N << std::endl
              << "Max particles per leaf: " << max_per_leaf << std::endl
              << "Number of iterations:   " << N_iter << std::endl
              << "Running on device:      " << device_ID
                                            << " (" << deviceProp.name << ")"
                                            << std::endl
              << std::endl;


    grace::Tree d_tree(N, max_per_leaf);
    build_tree(d_spheres, d_tree);

    // Ray origin is the box centre.
    grace::Vector<3, float> origin, AABB_bot, AABB_top;
    grace::min_vec3(d_spheres, &AABB_bot);
    grace::max_vec3(d_spheres, &AABB_top);
    origin.x = (AABB_bot.x + AABB_top.x) / 2.;
    origin.y = (AABB_bot.y + AABB_top.y) / 2.;
    origin.z = (AABB_bot.z + AABB_top.z) / 2.;

    CUDATimer timer;
    double t_genray_nosort, t_cum_nosort, t_hit_nosort, t_all;
    double t_genray_dirkey, t_cum_dirkey, t_hit_dirkey;
    double t_genray_endkey, t_cum_endkey, t_hit_endkey;
    t_genray_nosort = t_cum_nosort = t_hit_nosort = t_all = 0.0;
    t_genray_dirkey = t_cum_dirkey = t_hit_dirkey = 0.0;
    t_genray_endkey = t_cum_endkey = t_hit_endkey = 0.0;
    for (int i = -1; i < N_iter; ++i)
    {
        timer.start();

        thrust::device_vector<grace::Ray> d_rays(N);
        thrust::device_vector<int> d_hit_counts_nosort(N);
        thrust::device_vector<int> d_hit_counts_dirkey(N);
        thrust::device_vector<int> d_hit_counts_endkey(N);
        thrust::device_vector<float> d_integrals(N);
        // Don't include above memory allocations in t_genray_nosort.
        timer.split();

        // No sorting of rays.
        // This is typically fastest for Gadget data because particles are
        // already sorted by their Hilbert keys, which is a better spatial
        // locality sort than is typically achieved with Morton keys, as is
        // done by grace.
        grace::one_to_many_rays(d_rays, origin, d_spheres, grace::NoSort);
        if (i >= 0) t_genray_nosort += timer.split();

        grace::trace_cumulative_sph(d_rays,
                                    d_spheres,
                                    d_tree,
                                    d_integrals);
        if (i >= 0) t_cum_nosort += timer.split();

        // Profiling the pure hit-count tracing is useful for optimizing the
        // traversal algorithm.
        grace::trace_hitcounts_sph(d_rays,
                                   d_spheres,
                                   d_tree,
                                   d_hit_counts_nosort);
        if (i >= 0) t_hit_nosort += timer.split();


        // Direction-based sort.

        grace::one_to_many_rays(d_rays, origin.x, origin.y, origin.z,
                                d_spheres, grace::DirectionSort);
        if (i >= 0) t_genray_dirkey += timer.split();

        grace::trace_cumulative_sph(d_rays,
                                    d_spheres,
                                    d_tree,
                                    d_integrals);
        if (i >= 0) t_cum_dirkey += timer.split();

        grace::trace_hitcounts_sph(d_rays,
                                   d_spheres,
                                   d_tree,
                                   d_hit_counts_dirkey);
        if (i >= 0) t_hit_dirkey += timer.split();


        // Ray end-point sort.

        grace::one_to_many_rays(d_rays, origin.x, origin.y, origin.z,
                                d_spheres, grace::EndPointSort);
        // Using the already-computed AABBs is supported by GRACE, and would
        // be faster here; note the lack of grace::EndPointSort, which is
        // implicit:
        // grace::one_to_many_rays(d_rays, origin.x, origin.y, origin.z,
        //                         d_spheres, AABB_bot, AABB_top);
        if (i >= 0) t_genray_endkey += timer.split();

        grace::trace_cumulative_sph(d_rays,
                                    d_spheres,
                                    d_tree,
                                    d_integrals);
        if (i >= 0) t_cum_endkey += timer.split();

        grace::trace_hitcounts_sph(d_rays,
                                   d_spheres,
                                   d_tree,
                                   d_hit_counts_endkey);
        if (i >= 0) t_hit_endkey += timer.split();

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
            trace_bytes += d_integrals.size() * sizeof(float);
            trace_bytes += grace::N_table * sizeof(double); // Integral lookup.

            std::cout << "Memory allocated for cumulative trace kernel: "
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

    std::cout << "Time for generating unsorted rays:              "
              << std::setw(8) << t_genray_nosort / N_iter << " ms" << std::endl
              << "Time for hit count tracing:                     "
              << std::setw(8) << t_hit_nosort / N_iter << " ms" << std::endl
              << "Time for cumulative density tracing:            "
              << std::setw(8) << t_cum_nosort / N_iter << " ms" << std::endl
              << std::endl
              << "Time for generating and direction-sorting rays: "
              << std::setw(8) << t_genray_dirkey / N_iter << " ms" << std::endl
              << "Time for hit count tracing:                     "
              << std::setw(8) << t_hit_dirkey / N_iter << " ms" << std::endl
              << "Time for cumulative density tracing:            "
              << std::setw(8) << t_cum_dirkey / N_iter << " ms" << std::endl
              << std::endl
              << "Time for generating and end point-sorting rays: "
              << std::setw(8) << t_genray_endkey / N_iter << " ms" << std::endl
              << "Time for hit count tracing:                     "
              << std::setw(8) << t_hit_endkey / N_iter << " ms" << std::endl
              << "Time for cumulative density tracing:            "
              << std::setw(8) << t_cum_endkey / N_iter << " ms" << std::endl
              << std::endl
              << "Time for total (inc. memory ops):               "
              << std::setw(8) << t_all / N_iter << " ms" << std::endl
              << std::endl;

} // End device code.

    // Exit cleanly to ensure full profiler (nvprof/nvvp) trace.
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
