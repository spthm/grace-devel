#pragma once

#include "AABB.cuh"
#include "ray.cuh"

__host__ __device__ int plucker(const Ray&, const AABB&);
