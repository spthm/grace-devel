#pragma once

#include "AABB.cuh"
#include "ray.cuh"

struct WilliamsRayAuxillary
{
    float invdx, invdy, invdz;
};

__host__ __device__ WilliamsRayAuxillary williams_auxillary(const Ray&);


__host__ __device__ int williams(const Ray&,
                                 const WilliamsRayAuxillary&,
                                 const AABB&);

__host__ __device__ int williams_noif(const Ray&,
                                      const WilliamsRayAuxillary&,
                                      const AABB&);
