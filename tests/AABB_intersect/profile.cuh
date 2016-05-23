#pragma once

#include "AABB.cuh"
#include "ray.cuh"

#include "grace/error.h" // GRACE

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <time.h>

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
    #pragma omp parallel for
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

    const int num_blocks = std::min((rays.size() + 127) / 128, (size_t)48);

    cudaEventRecord(start);
    intersect_kernel<<<num_blocks, 128>>>(
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
    // We could do something like
    //   #ifdef _OPENMP
    //   float t = omp_get_wtime()
    //   #else
    //   struct timespec start;
    //   ...
    //   #endif
    // But this keeps the timing method consistent regardless of OpenMP support.
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    intersect(rays, boxes, hits, intersector);

    clock_gettime(CLOCK_MONOTONIC, &end);
    // Elapsed time in ms to match cudaEventElapsedTime().
    float elapsed = 1e3 * (end.tv_sec - start.tv_sec)
                      + 1e-6 * (end.tv_nsec - start.tv_nsec);
    return elapsed;
}
