#pragma once

#include "AABB.cuh"
#include "ray.cuh"

__host__ __device__ int williams(const Ray&, const AABB&);

__host__ __device__ int williams_noif(const Ray&, const AABB&);
