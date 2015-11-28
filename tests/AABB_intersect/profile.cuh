#pragma once

#include "AABB.cuh"
#include "ray.cuh"

#include "error.h" // GRACE

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdexcept>
#include <string>

template <typename Intersector>
__global__ void intersect_kernel(const Ray* rays, const AABB* boxes,
                                 int* hits,
                                 const size_t N_rays,
                                 const size_t N_AABBs,
                                 const Intersector intersector)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        // While its impact varies with the number of intersections per ray, it
        // still seems fairer to include any per-ray overhead in the
        // profiling results.
        Ray ray = intersector.prepare(rays[tid]);

        int hit_count = 0;
        for (size_t i = 0; i < N_AABBs; ++i)
        {
            AABB box = boxes[i];
            hit_count += intersector.intersect(ray, box);
        }

        hits[tid] = hit_count;
        tid += blockDim.x * gridDim.x;
    }
}

template <typename Intersector>
void intersect(const thrust::host_vector<Ray>& rays,
               const thrust::host_vector<AABB>& boxes,
               thrust::host_vector<int>& hits,
               const Intersector intersector)
{
    for (size_t i = 0; i < rays.size(); ++i)
    {
        Ray ray = intersector.prepare(rays[i]);
        int hit_count = 0;

        for (size_t j = 0; j < boxes.size(); ++j)
        {
            AABB box = boxes[j];
            hit_count += intersector.intersect(ray, box);
        }

        hits[i] = hit_count;
    }
}

// Inline only because it's defined in a header file.
inline int compare_hitcounts(const thrust::host_vector<int>& hits_a,
                             const std::string name_a,
                             const thrust::host_vector<int>& hits_b,
                             const std::string name_b,
                             const bool verbose = true)
{
    if (hits_a.size() != hits_b.size()) {
        throw std::invalid_argument("Hit vectors are different sizes");
    }

    size_t errors = 0;
    for (size_t i = 0; i < hits_a.size(); ++i)
    {
        if (hits_a[i] != hits_b[i]) {
            ++errors;

            if (!verbose) {
                continue;
            }

            std::cout << "Ray " << i << ": " << name_a << " != " << name_b
                      << "  (" << hits_a[i] << " != " << hits_b[i] << ")"
                      << std::endl;
        }
    }
    if (errors != 0) {
        std::cout << name_a << " != " << name_b << " (" << errors << " case"
                  << (errors > 1 ? "s)" : ")") << std::endl;
    }
    else {
        std::cout << name_a << " == " << name_b << " for all ray-AABB pairs"
                  << std::endl;
    }
    if (verbose) {
        std::cout << std::endl;
    }

    return errors;
}

template <typename Intersector>
float profile_gpu(const thrust::device_vector<Ray>& rays,
                  const thrust::device_vector<AABB>& boxes,
                  thrust::device_vector<int>& hits,
                  const Intersector intersector)
{
    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    intersect_kernel<<<48, 128>>>(
        thrust::raw_pointer_cast(rays.data()),
        thrust::raw_pointer_cast(boxes.data()),
        thrust::raw_pointer_cast(hits.data()),
        rays.size(),
        boxes.size(),
        intersector);
    cudaEventRecord(stop);
    GRACE_KERNEL_CHECK();
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    return elapsed;
}

template <typename Intersector>
float profile_cpu(const thrust::host_vector<Ray>& rays,
                  const thrust::host_vector<AABB>& boxes,
                  thrust::host_vector<int>& hits,
                  const Intersector intersector)
{
    clock_t t = clock();
    intersect(rays, boxes, hits, intersector);
    float elapsed = ((double) (clock() - t)) / CLOCKS_PER_SEC;

    return (float)(1000 * t); // convert to ms.
}
