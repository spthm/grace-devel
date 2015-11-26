#pragma once

#include "AABB.cuh"
#include "ray.cuh"

__host__ __device__ int eisemann(const Ray&, const AABB&);
