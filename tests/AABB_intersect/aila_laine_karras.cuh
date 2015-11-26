#pragma once

#include "AABB.cuh"
#include "ray.cuh"

__device__ int aila_laine_karras(const Ray&, const AABB&);
