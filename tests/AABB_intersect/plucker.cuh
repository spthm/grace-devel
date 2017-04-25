#pragma once

#include "AABB.cuh"
#include "ray.cuh"

struct PluckerRayAuxillary
{
    unsigned int dclass;
};

__host__ __device__ PluckerRayAuxillary plucker_auxillary(const Ray& ray);
__host__ __device__ int plucker(const Ray& ray,
                                const PluckerRayAuxillary& aux,
                                const AABB& box);
