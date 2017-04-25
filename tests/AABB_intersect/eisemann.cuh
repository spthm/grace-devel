#pragma once

#include "AABB.cuh"
#include "ray.cuh"

struct EisemannRayAuxillary
{
    unsigned int dclass;
    float xbyy, ybyx, ybyz, zbyy, xbyz, zbyx;
    float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
};

__host__ __device__ EisemannRayAuxillary eisemann_auxillary(const Ray&);
__host__ __device__ int eisemann(const Ray&,
                                 const EisemannRayAuxillary&,
                                 const AABB&);
