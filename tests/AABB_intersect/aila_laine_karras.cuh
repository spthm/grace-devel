#pragma once

#include "AABB.cuh"
#include "ray.cuh"

struct AilaRayAuxillary
{
    float invdx, invdy, invdz;
};

__host__ __device__ AilaRayAuxillary aila_auxillary(const Ray&);
__device__ int aila(const Ray&, const AilaRayAuxillary&, const AABB&);
